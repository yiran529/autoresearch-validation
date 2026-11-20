# n020-engineer-maxu

Implementation: `72a57d8aa3e1c5b85fcc0e5240c353fd97c77363`
Parameter source: `112bf8c8fa478547c8cfe37aec8224a00acc2cd5`

DDP gradient compression before AdamW. Public registration and method sources are reused unchanged. Communication module overlay includes all required sparse CUDA helpers; no candidate optimizer or training loop is transplanted. Optional diagnostic checkpoint branch remains disabled.

Key source assignments are compared by AST against the maximum stored-U version. N020 requested rank32 is internally clamped to effective rank8 under EF14; K10 and period80 are retained. N023 keeps 8/256 and N031 keeps 128/4096. GreedyLoRE uses N000 Engineer rank32/gap200.

Only LLaMA supported in this campaign. Static review is complete; runtime evidence is saved with the real-model smoke/experiment.
