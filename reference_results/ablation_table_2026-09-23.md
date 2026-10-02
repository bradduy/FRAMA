## Leave-one-out from the equal six-agent stack, official CIDEr / SPICE (delta vs six-agent)

| Removed agent | COCO | NoCaps overall | NoCaps out | Flickr30k |
|---|---|---|---|---|
| none (six-agent) | 120.8 / 22.2 | 81.3 / 11.0 | 79.1 / 9.9 | 70.5 / 18.2 |
| minus LM | 120.5 / 22.2 (-0.3) | 82.3 / 11.0 (+1.0) | 79.6 / 9.9 (+0.5) | 72.1 / 18.5 (+1.6) |
| minus bigG | 119.2 / 22.0 (-1.6) | 80.1 / 10.8 (-1.2) | 77.5 / 9.8 (-1.6) | 68.6 / 18.0 (-1.9) |
| minus BLIP | 118.9 / 21.9 (-1.9) | 79.5 / 10.7 (-1.8) | 77.2 / 9.7 (-1.9) | 69.2 / 17.9 (-1.3) |
| minus VQA | 119.2 / 22.0 (-1.6) | 80.2 / 10.8 (-1.1) | 77.5 / 9.7 (-1.6) | 69.7 / 18.3 (-0.8) |
| minus MBR | 119.5 / 22.1 (-1.3) | 81.5 / 11.0 (+0.2) | 79.1 / 9.9 (+0.0) | 71.4 / 18.3 (+0.9) |
| minus ground | 120.4 / 22.1 (-0.4) | 80.7 / 10.9 (-0.6) | 78.8 / 10.0 (-0.3) | 69.6 / 18.1 (-0.9) |

## Weight transfer: best Dirichlet-sweep weights of one benchmark applied to the others, official CIDEr (delta vs equal weights)

| Weights fitted on | COCO | NoCaps overall | NoCaps out | Flickr30k |
|---|---|---|---|---|
| coco | in-sample, not reported | 82.4 (+1.1) | 80.8 (+1.7) | 71.9 (+1.4) |
| nocaps | 119.3 (-1.5) | in-sample, not reported | in-sample, not reported | 71.4 (+0.9) |
