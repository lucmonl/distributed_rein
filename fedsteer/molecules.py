"""Molecule attribute and quality helpers (ChEMBL task).

The attribute is the *decoration lipophilicity*

    a(y) = cLogP(y) - cLogP(MurckoScaffold(y))

computed on the generated molecule's own scaffold, exactly as the training labels
are computed. Raw descriptors are ~85% determined by the scaffold that is given in
the prompt, which would leave almost nothing to steer; the residual is the part the
model is actually free to choose (see chembl-experiment-plan.md §2.2).

Unparseable output scores ``nan``; callers report validity separately and compute
steering metrics on the valid outputs only.
"""

from __future__ import annotations

import re
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

NAN = float("nan")
# the model may wrap the answer in prose or a code fence; take the first
# whitespace-free run of SMILES-legal characters that RDKit accepts
_CAND = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$%/\\\.:*]{3,}")


def extract_smiles(text: str) -> Optional[str]:
    """First substring of ``text`` that parses as a molecule, or None."""
    text = text.strip().strip("`")
    for cand in [text] + _CAND.findall(text):
        if Chem.MolFromSmiles(cand) is not None:
            return cand
    return None


def mol_of(text: str):
    smi = extract_smiles(text)
    return Chem.MolFromSmiles(smi) if smi else None


def canonical(text: str) -> Optional[str]:
    m = mol_of(text)
    return Chem.MolToSmiles(m) if m is not None else None


def scaffold_of(mol):
    try:
        s = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None
    return s if s is not None and s.GetNumHeavyAtoms() > 0 else None


_CORE_LOGP: dict[str, float] = {}


def core_clogp(scaffold_smiles: str) -> float:
    """cLogP of a core, cached (an eval asks for the same cores at every alpha)."""
    if scaffold_smiles not in _CORE_LOGP:
        core = Chem.MolFromSmiles(scaffold_smiles)
        _CORE_LOGP[scaffold_smiles] = NAN if core is None else float(Crippen.MolLogP(core))
    return _CORE_LOGP[scaffold_smiles]


def clogp_residual(text: str, rec: dict) -> float:
    """The attribute of a generated molecule: cLogP(y) - cLogP(**the core given in
    the prompt**).

    The baseline is the *requested* scaffold, never the generated molecule's own
    Murcko scaffold. With a self-referential baseline a model that swapped the core
    would move the very quantity being subtracted, so it could land on any residual
    without decorating what it was asked to decorate -- the attribute would be
    partly under the model's control instead of being a property of its answer.

    On training targets the two definitions coincide exactly, because the prompt's
    core IS that target's Murcko scaffold by construction (scripts/build_chembl_fed.py),
    so the training labels are unchanged; the definitions diverge only where a
    generation abandons the core.

    Abandoning the core is still a *quality* failure, and a visible one: report it
    with ``keeps_scaffold`` / the retention metric, and cross-check conclusions with
    ``clogp_residual_strict``. This mirrors how the Newsroom task reads density next
    to a faithfulness metric rather than redefining density.

    ``nan`` when the generation does not parse; such rows are dropped by
    ``metrics_for_client`` and counted in ``unscorable_row_rate``.
    """
    m = mol_of(text)
    if m is None:
        return NAN
    return float(Crippen.MolLogP(m) - core_clogp(rec["scaffold"]))


def clogp_residual_strict(text: str, rec: dict) -> float:
    """As ``clogp_residual``, but ``nan`` unless the generated molecule still
    contains the requested core -- a non-decoration is not a solution to the task.
    Used as a robustness check on the primary metric, not as the primary metric:
    it drops a whole prompt when any single alpha loses the core."""
    if not keeps_scaffold(text, rec["scaffold"]):
        return NAN
    return clogp_residual(text, rec)


def clogp_residual_self(text: str, rec: Optional[dict] = None) -> float:
    """Diagnostic only: residual against the generation's *own* Murcko scaffold.
    Kept to quantify how far a run's outputs drift from the requested core; never
    use it as the attribute (see ``clogp_residual``)."""
    m = mol_of(text)
    if m is None:
        return NAN
    s = scaffold_of(m)
    return NAN if s is None else float(Crippen.MolLogP(m) - Crippen.MolLogP(s))


def keeps_scaffold(text: str, requested_scaffold: str) -> Optional[bool]:
    """Did the generated molecule keep the core it was asked to decorate?
    None if either side does not parse."""
    m = mol_of(text)
    core = Chem.MolFromSmiles(requested_scaffold) if requested_scaffold else None
    if m is None or core is None:
        return None
    return bool(m.HasSubstructMatch(core))


def descriptors(text: str) -> dict:
    """Off-target and quality descriptors of a generated molecule."""
    m = mol_of(text)
    if m is None:
        return {"valid": 0.0}
    return {
        "valid": 1.0,
        "mw": float(rdMolDescriptors.CalcExactMolWt(m)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(m)),
        "clogp": float(Crippen.MolLogP(m)),
        "qed": float(QED.qed(m)),
        "heavy_atoms": float(m.GetNumHeavyAtoms()),
        "rings": float(rdMolDescriptors.CalcNumRings(m)),
    }


# --------------------------------------------------------------------------- #
# Decoration format (Arus-Pous et al. 2020; the same prefix convention SAFE uses
# for scaffold decoration).  The model is given the core with numbered attachment
# points and emits ONLY the decorations, so it can never rewrite or renumber the
# core: retention becomes structural instead of learned.  Entry 33/34 showed the
# whole-molecule format loses the core on 42% of generations, rising with alpha.
# --------------------------------------------------------------------------- #

_MOLZIP = None


def _molzip_params():
    global _MOLZIP
    if _MOLZIP is None:
        p = Chem.MolzipParams()
        p.label = Chem.MolzipLabel.Isotope
        _MOLZIP = p
    return _MOLZIP


def split_core_decorations(smiles: str, core_smiles: str):
    """(core with [n*] attachment points, '.'-joined decorations) or None.

    Both sides of every cut bond carry the *same* dummy label, so the pieces can
    be zipped back together unambiguously.
    """
    mol = Chem.MolFromSmiles(smiles)
    core = Chem.MolFromSmiles(core_smiles)
    if mol is None or core is None:
        return None
    match = mol.GetSubstructMatch(core)
    if not match:
        return None
    inside = set(match)
    bonds = [b.GetIdx() for b in mol.GetBonds()
             if (b.GetBeginAtomIdx() in inside) != (b.GetEndAtomIdx() in inside)]
    if not bonds:
        return None
    labels = [(i + 1, i + 1) for i in range(len(bonds))]
    try:
        frag = Chem.FragmentOnBonds(mol, bonds, addDummies=True, dummyLabels=labels)
        pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)
    except Exception:
        return None
    core_parts, deco_parts = [], []
    for p in pieces:
        try:
            (core_parts if p.HasSubstructMatch(core) else deco_parts).append(Chem.MolToSmiles(p))
        except Exception:
            return None
    if len(core_parts) != 1 or not deco_parts:
        return None
    return core_parts[0], ".".join(sorted(deco_parts))


def rejoin_decorations(core_attached: str, decorations: str):
    """Zip generated decorations onto the core from the prompt. None if the
    decorations are malformed or their attachment labels do not match the core."""
    a = Chem.MolFromSmiles(core_attached, sanitize=False)
    b = Chem.MolFromSmiles(decorations, sanitize=False)
    if a is None or b is None:
        return None
    try:
        m = Chem.molzip(a, b, _molzip_params())
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def extract_decorations(text: str) -> Optional[str]:
    """Pull the decoration string out of a generation (first line, fence-stripped)."""
    t = text.strip().strip("`").split("\n")[0].strip()
    return t or None


def clogp_residual_deco(text: str, rec: dict) -> float:
    """Attribute in the decoration format: cLogP(core + generated decorations)
    - cLogP(the clean core given in the prompt).

    The core comes from the prompt, so the baseline cannot drift (entry 32) *and*
    the core cannot be lost (entry 34). ``nan`` if the decorations do not zip on.
    """
    deco = extract_decorations(text)
    if not deco:
        return NAN
    m = rejoin_decorations(rec["core_attached"], deco)
    if m is None:
        return NAN
    return float(Crippen.MolLogP(m) - core_clogp(rec["scaffold"]))


def assembled_descriptors(text: str, rec: dict) -> dict:
    """Validity / descriptors of the molecule a generation assembles to."""
    deco = extract_decorations(text)
    m = rejoin_decorations(rec["core_attached"], deco) if deco else None
    if m is None:
        return {"valid": 0.0}
    out = {"valid": 1.0, "smiles": Chem.MolToSmiles(m),
           "mw": float(rdMolDescriptors.CalcExactMolWt(m)),
           "tpsa": float(rdMolDescriptors.CalcTPSA(m)),
           "clogp": float(Crippen.MolLogP(m)),
           "qed": float(QED.qed(m)),
           "heavy_atoms": float(m.GetNumHeavyAtoms()),
           "rings": float(rdMolDescriptors.CalcNumRings(m))}
    core = Chem.MolFromSmiles(rec["scaffold"])
    out["keeps_core"] = float(bool(core is not None and m.HasSubstructMatch(core)))
    return out
