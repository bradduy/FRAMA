# E6 results against the pre-registration (2026-10-27)

Run on GCP2 03:00 to 07:22 UTC; analysis `e6_analyze.py`, raw numbers `e6_results.json`, console `logs/e6_analyze.out`.
Two tests were added after the first look at the numbers and are not pre-registered: the paired difference of Kendall tau between the verifier and CHAIR or SPICE, and the paired difference of the five-agent sum against each single agent on E6b.
Intervals: paired bootstrap, 10,000 resamples over images (E6a, E6c) or items (E6b), 2,000 image resamples for Kendall tau.

## E6c validation (gate)

THumB 1.0, 2,000 machine captions, caption-level Kendall tau with human Precision:

| Instrument | tau | pairwise within-image agreement |
|---|---|---|
| verifier share of supported facts | 0.395 [0.346, 0.440] | 0.708 |
| CHAIR_s (negated) | 0.301 [0.249, 0.348] | 0.568 |
| SPICE precision vs references | 0.224 [0.179, 0.268] | 0.651 |

Paired difference verifier minus CHAIR +0.094 [+0.034, +0.157]; verifier minus SPICE +0.171 [+0.122, +0.221].
Expectation tau >= 0.25 and above CHAIR: **met**. Gate (tau < 0.15 = inconclusive) passed.

## E6a: SPICE by tuple category

| Expectation | Result |
|---|---|
| six-agent precision above captioner for Object, Attribute, Relation on COCO | **met**: +3.9, +6.2, +1.8 points, all intervals exclude 0 |
| object recall within 1 point of the captioner | **met**, and higher: +1.0 [+0.8, +1.1] |
| attribute recall may fall by up to 1 point | against the captioner it rises (+0.7); against the z-sum it falls by 0.4 [-0.6, -0.3] |

The same pattern holds on NoCaps and Flickr30k (overall precision +5.9 and +3.2 for six agents against the captioner, recall +0.6 and +0.7).

## E6b: human-validated foils (1,198 items, 439 COCO Karpathy-test images)

| Expectation | Result |
|---|---|
| System One verifier best single agent on relation and attribute replacement | **not met**: BLIP-ITM is best on both (81.9 vs 77.7; 90.0 vs 89.0). The System One verifier is best on VALSE spatial relations (81.7 vs 68.3) and ties on add attribute |
| bigG weakest on swaps | **partly**: weakest of the three alignment agents on swap object (54.5); on swap attribute the System One verifier is lower (76.1 vs 80.3); grounding is at chance on swaps by construction |
| five-agent at least the best single agent on the macro mean, not on every phenomenon | **borderline**: macro 85.1 vs BLIP-ITM 85.3, difference -0.3 [-4.0, +3.6]; all items 90.1 vs 88.7, +1.3 [-0.6, +3.3]. Not on every phenomenon: existence 66.7 vs 91.7 (n = 12), counting 60.9 vs 67.4 |
| all agents near chance on counting | **mostly**: 48.9 to 60.9 except BLIP-ITM 67.4 |

Five-agent against the bigG + BLIP-ITM z-sum: +1.8 [+0.0, +3.6] on all items. Largest gains of the five-agent sum over the best single agent: replace relation 87.3 vs 81.9, add attribute 97.6 (tie with consensus), swap object 75.8 vs 72.7, VALSE spatial relations 83.3 vs 81.7.

## E6c: atomic facts of our own selections

| Expectation | Result |
|---|---|
| COCO six-agent has 15 to 35% fewer unsupported facts than the captioner | **exceeded**: 20.2 -> 12.9%, 36% relative, -7.3 [-7.8, -6.7] points |
| reduction holds for attributes and relations | **met**: attribute 25.6 -> 16.2, colour 37.6 -> 24.4, count 18.9 -> 8.9, spatial 30.4 -> 20.8, action 33.9 -> 22.8 |
| supported facts per caption not more than 5% below the captioner | **met**, higher: 4.71 -> 5.05, +0.34 [+0.30, +0.37] |

Against the Borda proxy of v1: -1.1 [-1.5, -0.7] points on COCO and -1.6 on NoCaps (six agents); five agents -2.1 and -3.0.
Against the three-agent z-sum: six agents -0.6 [-1.0, -0.2] on COCO, -0.2 [-0.6, +0.3] on NoCaps, -0.1 [-0.9, +0.7] on Flickr30k; five agents -1.6 on COCO and NoCaps (both significant), -0.6 [-1.4, +0.3] on Flickr30k. Supported facts per caption fall by 0.04 for six agents against the z-sum.

## Reading

Three instruments with different ground truth (reference scene graphs, human-validated foils, an image-grounded verifier validated against human precision) agree that the selections of FRAMA reduce factual errors beyond objects (attributes, colour, counts, relations, actions) against the captioner, without dropping content. The increment of the added agents over the three-agent z-sum is small and significant on COCO, and on NoCaps only for the five-agent configuration. On foils the five-agent sum is not distinguishable from BLIP-ITM alone; its value there is on relations and attribute binding, and it is weak on existence and counting.
