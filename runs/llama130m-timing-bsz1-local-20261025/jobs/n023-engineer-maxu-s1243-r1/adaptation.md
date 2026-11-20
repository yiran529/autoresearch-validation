# n023-engineer-maxu

Implementation: `63205995a2c0380e7bab2c254a001df3306741bd`
Parameter source: `63205995a2c0380e7bab2c254a001df3306741bd`

DDP gradient compression before AdamW. Public registration and method sources are reused unchanged. Communication module overlay includes all required sparse CUDA helpers; no candidate optimizer or training loop is transplanted. Optional diagnostic checkpoint branch remains disabled.

Key source assignments are compared by AST against the maximum stored-U version. N020 requested rank32 is internally clamped to effective rank8 under EF14; K10 and period80 are retained. N023 keeps 8/256 and N031 keeps 128/4096. GreedyLoRE uses N000 Engineer rank32/gap200.

Only LLaMA supported in this campaign. Static review is complete; runtime evidence is saved with the real-model smoke/experiment.
