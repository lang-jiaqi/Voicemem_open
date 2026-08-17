# Evaluation: AudioMC

## Protocol

- Benchmark: AudioMC `INFERENCE_MEMORY`
- Evaluation set: 132 conversations, 233 rubric criteria
- Answer model: one model per complete run
- Judge: `gpt-4o-mini`
- Metric: satisfied rubric criteria / total rubric criteria
- The same prompts, memory setup, judge, and rubric must be used for every compared model.

## Recorded baseline

| Answer model | Satisfied criteria | Total | Score |
| --- | ---: | ---: | ---: |
| GPT-4o-mini | 96 | 233 | 41.2% |
| Voicemem QLoRA checkpoint-3318 | 97 | 233 | 41.6% |

The Voicemem adapter improves over the GPT-4o-mini baseline by 1 criterion out of 233 (+0.43 percentage points). The baseline and checkpoint-3318 result use the same 132 conversations, rubric, and GPT-4o-mini judge.

Archived checkpoint-3318 result SHA256:

```text
0285c0682bcea08905b0138453561f63bb9e06dc08855498687841613a551e5a
```

## Reporting requirements

- State the exact answer model, judge model, model revision/date, and decoding parameters.
- Do not compare a locally served model with an API baseline unless the judge and test protocol are identical.
- Report failures and incomplete samples, if any.
- AudioMC results measure this benchmark only; they are not a general safety or capability claim.
