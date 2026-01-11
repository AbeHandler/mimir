from mimir.attacks.all_attacks import AllAttacks

from mimir.attacks.loss import LOSSAttack
from mimir.attacks.loss_rank_clipped import LOSSRankClippedAttack
from mimir.attacks.loss_rank_skipped import LOSSRankSkippedAttack
from mimir.attacks.loss_bisection import LOSSBisectionAttack
from mimir.attacks.reference import ReferenceAttack
from mimir.attacks.zlib import ZLIBAttack
from mimir.attacks.min_k import MinKProbAttack
from mimir.attacks.min_k_plus_plus import MinKPlusPlusAttack
from mimir.attacks.neighborhood import NeighborhoodAttack
from mimir.attacks.gradnorm import GradNormAttack
from mimir.attacks.recall import ReCaLLAttack
#from mimir.attacks.dc_pdd import DC_PDDAttack


# TODO Use decorators to link attack implementations with enum above
def get_attacker(attack: str):
    mapping = {
        AllAttacks.LOSS: LOSSAttack,
        AllAttacks.LOSS_RANK_CLIPPED: LOSSRankClippedAttack,
        AllAttacks.LOSS_RANK_SKIPPED: LOSSRankSkippedAttack,
        AllAttacks.LOSS_BISECTION: LOSSBisectionAttack,
        AllAttacks.REFERENCE_BASED: ReferenceAttack,
        AllAttacks.ZLIB: ZLIBAttack,
        AllAttacks.MIN_K: MinKProbAttack,
        AllAttacks.MIN_K_PLUS_PLUS: MinKPlusPlusAttack,
        AllAttacks.NEIGHBOR: NeighborhoodAttack,
        AllAttacks.GRADNORM: GradNormAttack,
        AllAttacks.RECALL: ReCaLLAttack,
        #AllAttacks.DC_PDD: DC_PDDAttack
    }
    attack_cls = mapping.get(attack, None)
    if attack_cls is None:
        raise ValueError(f"Attack {attack} not found")
    return attack_cls
