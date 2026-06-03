# Vendored third-party code

This directory contains adapted copies of third-party topic-modeling libraries,
kept here so the repository is self-contained and the baselines reproduce exactly.
All credit goes to the original authors; please cite them if you use these baselines.

| Subpackage | Upstream project | Provides (baselines used here) |
|------------|------------------|--------------------------------|
| `octis/`   | [OCTIS](https://github.com/MIND-Lab/OCTIS) (Terragni et al., 2021) | ProdLDA, CombinedTM, ZeroShotTM (`CTM`), ETM, LDA |
| `topmost/` | [TopMost](https://github.com/BobXWu/TopMost) (Wu et al.)            | ECRTM (Wu et al., ICML 2023) |
| `fastopic/`| [FASTopic](https://github.com/BobXWu/FASTopic) (Wu et al., 2024)    | FASTopic |

Only thin edges are adapted (e.g. import paths). The authors' own contribution — the
DSL training objective and its model variants — lives in `dsl_topic/models/dsl/`, not here.
