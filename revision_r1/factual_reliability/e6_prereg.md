# E6: factual reliability beyond CHAIR (R2 W4, F6, S4)

Written 2026-10-27 before any E6 number was computed. Results that miss these expectations are reported as they are.

## Selectors

Fixed configurations of the paper, on the stored beams, no new selection rule:
captioner top-1 (`lm_top1`), Borda proxy of the published baseline (`borda3_lm_bigG_blip`), three-agent z-sum
(`zsum3_lm_bigG_blip`), six-agent FRAMA (`stack6_all_equal`), five-agent FRAMA without LM
(`stack5_noLM_blip_bigG_vqa_mbr_ground`). Benchmarks: COCO Karpathy test (5,000), NoCaps val (4,500), Flickr30k test (1,000).
Main comparison: six-agent against the captioner on COCO and Flickr30k, five-agent against the captioner on NoCaps; also against the Borda proxy and the z-sum.

## E6a: SPICE by tuple category

SPICE 1.0 with `-detailed -subset`, official jar, Java 11, same references as the paper. For each category
(Object, Attribute, Relation, Color, Count = Cardinality, Size) we pool true positives, false positives and false negatives over
images and report precision, recall and F; paired bootstrap over images, 10,000 resamples, seed 0.

Expectations:
- Six-agent precision is above the captioner's for Object, Attribute and Relation on COCO.
- Object recall is within 1 point of the captioner's. The question is whether the gain comes from omitting content; we expect it does not, because object coverage is unchanged (43.5 vs 42.7%).
- Attribute recall may fall by up to 1 point. THumB human recall fell by 0.11 for the added agents.

## E6b: human-validated foils

SugarCrepe (all seven subsets) and VALSE (plurals, relations, foil-it, counting-hard, counting-small-quant, counting-adversarial,
existence; only items that the majority of VALSE annotators validated). Restricted to images of the COCO Karpathy test split,
which no COCO-fine-tuned model saw in training and for which our beams exist. SWiG (actions) and VisDial (coreference) images are not COCO
and are excluded.

For each item the correct caption and its foil are added to the 20-candidate beam of the image. Every agent scores both, the
consensus agent is recomputed over the 22 candidates, and each agent is z-normalised within the 22. An item counts as correct
if the selector scores the correct caption above the foil; ties count one half. The LM agent is undefined for captions our
captioner did not produce, so the combined selector is the five-agent configuration, as in the THumB analysis.
Bootstrap over items.

Expectations:
- The System One verifier is the best single agent on relation and attribute replacement, and bigG is the weakest on swaps.
- The five-agent sum is at least as accurate as the best single agent averaged over phenomena, but not on every phenomenon.
- All agents are near chance on counting.

## E6c: atomic-fact verification of our own selections

The atomic facts of a caption are the SPICE scene-graph tuples of that caption, typed as object, attribute, colour, count,
size, spatial relation or action. A relation whose predicate is a preposition or a spatial phrase is spatial; any other relation is an action.
An attribute that is a verb form is an action. Each fact is checked against the image by Qwen3-VL-4B-Instruct, which is not one of the
FRAMA agents. The prompt contains the caption and the fact, and the read-out is P(yes) / (P(yes) + P(no)) on the first answer token.
A fact is supported if that value is above 0.5. The images are resized to at most 448 x 448.

Validation against humans, before any selector is compared: on the 2,000 THumB 1.0 machine captions, the caption-level share of supported facts
is correlated with the human Precision rating. It is compared with CHAIR and with SPICE precision against the references, and with the
within-image pairwise agreement with the human ordering.

Expectations:
- The verifier's share of supported facts correlates with human Precision at Kendall tau of at least 0.25, above CHAIR.
- On COCO the six-agent selection has fewer unsupported facts than the captioner, by 15 to 35 percent relative. The reduction holds for
  attributes and relations, not only objects.
- The number of supported facts per caption is not lower than the captioner's by more than 5 percent.

If the verifier fails the validation, for example with tau below 0.15, E6c is reported as inconclusive and not used for any claim.
