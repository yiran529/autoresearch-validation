# n031-engineer-maxu-resnet-v1

Parent: n031-engineer-maxu; source commit 8684c7acad3850ed029cf6039345b890cf674472.
Source files and frozen key parameters identical to parent. No new search selection.
Candidate native4D convolution flattening retained. Dense1D/BatchNorm and linear
classifier use existing eligibility, with linear explicitly excluded as in the canonical driver.
Adapter temporarily supplies ResNet classifier names during public registration,
then restores the module helper even on exceptions. Muon update and data pipeline unchanged.
Static support declaration only; synthetic205step and full40epoch evidence live in this campaign.
Nominal rank32/gap200 retained; N20 internally caps effective rank8 and rejuvenates every10.
