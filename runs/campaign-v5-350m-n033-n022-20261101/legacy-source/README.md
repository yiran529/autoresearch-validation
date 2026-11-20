#350M Muon N33 then N22

User authorized2026-11-01: only GPU4-7, serial N33 Research2410afc4f56b989b4040e0b9930011dfbf428102 then N22 Engineer8cec48b5e58e89139ba368bfaae98817a475502f, seed1243. Previous130M queue must finish first. No Dense/GreedyLoRE/N19/N27 run.

60k updates,warmup6000,matrixLR.005 (historical350M Dense-selected),scalarLR.001,batch128/device/global512,GA1,seq256,BF16,activationcheckpointing,clip1,WD0,cosineto.1,compressionstart1000,eval1000. W&Boffline; no model/optimizer checkpoints. Existing data repeat, no downloads. Candidate snapshots/adapters reused unchanged; identity/checks retained from130M campaign.

Both actual350M formalbatch8step smoke jobs with compressionstart2 must pass and cover compression before full queue starts. Short recipe is memory/path validation only, not full-quality or60k equivalence. All logs within scaffold. Fixed vendor/scaffold/environment and external projects unchanged.

status.json/controller.log; tmux v5-350m-n033-n022; full plan /home/wyr/greedylore-validation/runs/v5-350m-n033-n022-20261101. Interruption cannot resume exact optimizer state; retain evidence and use newattempt if needed.

## User revision
Remaining short tests skipped. Active plan /home/wyr/greedylore-validation/runs/v5-350m-n033-n022-rolling20k-20261101; rolling checkpoint at20k/40k/60k under /dev/shm/wyr_tmp/greedylore-validation/v5-350m-n033-n022-20261101/<method>/seed-1243/latest. Includes rank0model and allrank optimizer/scheduler/EF/predictor/RNG. Complete generation published before old removed. Exact data replay/resume not implemented. Runtime code frozen in new plan; original plan/short-test evidence retained. tmux v5-350m-rolling20k; log rolling-controller.log.
