## Low data (900 training stays): overall test MSE

| config | seeds | none | random:0.5 | random:0.7 | bursty:0.5 | variable:0.3 | variable:0.5 | stride:3 |
|---|---|---|---|---|---|---|---|---|
| p3b_low_plain | 3 | 0.0085 ± 0.0004 | 0.0168 ± 0.0004 | 0.0220 ± 0.0007 | 0.0132 ± 0.0005 | 0.0218 ± 0.0037 | 0.0283 ± 0.0047 | 0.0145 ± 0.0001 |
| p3b_low_hard_llm | 3 | 0.0083 ± 0.0001 | 0.0183 ± 0.0007 | 0.0256 ± 0.0003 | 0.0141 ± 0.0003 | 0.0217 ± 0.0022 | 0.0295 ± 0.0030 | 0.0153 ± 0.0009 |
| p3b_low_hard_est | 3 | 0.0084 ± 0.0002 | 0.0173 ± 0.0009 | 0.0241 ± 0.0025 | 0.0140 ± 0.0004 | 0.0197 ± 0.0032 | 0.0266 ± 0.0047 | 0.0147 ± 0.0012 |
| p3b_low_hard_hybrid | 3 | 0.0084 ± 0.0002 | 0.0189 ± 0.0013 | 0.0265 ± 0.0027 | 0.0143 ± 0.0009 | 0.0225 ± 0.0041 | 0.0307 ± 0.0059 | 0.0166 ± 0.0012 |
| p3b_low_hard_shuffled | 3 | 0.0083 ± 0.0004 | 0.0174 ± 0.0008 | 0.0246 ± 0.0031 | 0.0136 ± 0.0004 | 0.0220 ± 0.0057 | 0.0303 ± 0.0095 | 0.0144 ± 0.0000 |
| p3b_low_hard_identity | 3 | 0.0083 ± 0.0001 | 0.0191 ± 0.0009 | 0.0262 ± 0.0015 | 0.0142 ± 0.0002 | 0.0213 ± 0.0019 | 0.0287 ± 0.0027 | 0.0169 ± 0.0008 |
| p3b_low_hard_random | 3 | 0.0082 ± 0.0004 | 0.0180 ± 0.0020 | 0.0248 ± 0.0043 | 0.0138 ± 0.0011 | 0.0207 ± 0.0050 | 0.0285 ± 0.0081 | 0.0147 ± 0.0022 |

### Low data (900 training stays): MSE on the 33 variables with >= 1 parent in the llm graph

variables: Albumin, ALT, AST, Bilirubin, BUN, Creatinine, DiasABP, FiO2, GCS, HCO3, HCT, HR, K, Lactate, Mg, MAP, MechVent, Na, NIDiasABP, NIMAP, NISysABP, PaCO2, PaO2, pH, Platelets, RespRate, SaO2, SysABP, Temp, TroponinI, TroponinT, Urine, WBC

| config | seeds | none | random:0.5 | random:0.7 | bursty:0.5 | variable:0.3 | variable:0.5 | stride:3 |
|---|---|---|---|---|---|---|---|---|
| p3b_low_plain | 3 | 0.0088 ± 0.0003 | 0.0179 ± 0.0002 | 0.0236 ± 0.0005 | 0.0141 ± 0.0003 | 0.0226 ± 0.0038 | 0.0289 ± 0.0049 | 0.0151 ± 0.0005 |
| p3b_low_hard_llm | 3 | 0.0087 ± 0.0001 | 0.0197 ± 0.0009 | 0.0277 ± 0.0004 | 0.0151 ± 0.0004 | 0.0225 ± 0.0026 | 0.0304 ± 0.0036 | 0.0164 ± 0.0011 |
| p3b_low_hard_est | 3 | 0.0088 ± 0.0002 | 0.0186 ± 0.0010 | 0.0260 ± 0.0028 | 0.0151 ± 0.0004 | 0.0203 ± 0.0038 | 0.0272 ± 0.0057 | 0.0158 ± 0.0013 |
| p3b_low_hard_hybrid | 3 | 0.0088 ± 0.0003 | 0.0205 ± 0.0013 | 0.0288 ± 0.0029 | 0.0154 ± 0.0009 | 0.0235 ± 0.0046 | 0.0318 ± 0.0067 | 0.0179 ± 0.0013 |
| p3b_low_hard_shuffled | 3 | 0.0087 ± 0.0005 | 0.0189 ± 0.0008 | 0.0268 ± 0.0034 | 0.0146 ± 0.0004 | 0.0230 ± 0.0064 | 0.0317 ± 0.0105 | 0.0155 ± 0.0001 |
| p3b_low_hard_identity | 3 | 0.0087 ± 0.0001 | 0.0206 ± 0.0009 | 0.0284 ± 0.0016 | 0.0152 ± 0.0001 | 0.0222 ± 0.0023 | 0.0297 ± 0.0032 | 0.0182 ± 0.0008 |
| p3b_low_hard_random | 3 | 0.0087 ± 0.0003 | 0.0196 ± 0.0022 | 0.0269 ± 0.0047 | 0.0149 ± 0.0011 | 0.0216 ± 0.0056 | 0.0295 ± 0.0091 | 0.0158 ± 0.0023 |

### Low data (900 training stays): MSE on the 12 variables with >= 1 parent in the estimated graph

variables: DiasABP, FiO2, GCS, MAP, NIDiasABP, NIMAP, NISysABP, PaCO2, PaO2, SysABP, Temp, Urine

| config | seeds | none | random:0.5 | random:0.7 | bursty:0.5 | variable:0.3 | variable:0.5 | stride:3 |
|---|---|---|---|---|---|---|---|---|
| p3b_low_plain | 3 | 0.0097 ± 0.0002 | 0.0217 ± 0.0014 | 0.0323 ± 0.0016 | 0.0144 ± 0.0002 | 0.0370 ± 0.0103 | 0.0493 ± 0.0141 | 0.0246 ± 0.0026 |
| p3b_low_hard_llm | 3 | 0.0093 ± 0.0002 | 0.0241 ± 0.0012 | 0.0396 ± 0.0025 | 0.0150 ± 0.0006 | 0.0368 ± 0.0081 | 0.0530 ± 0.0113 | 0.0276 ± 0.0018 |
| p3b_low_hard_est | 3 | 0.0097 ± 0.0003 | 0.0234 ± 0.0021 | 0.0380 ± 0.0073 | 0.0162 ± 0.0009 | 0.0315 ± 0.0109 | 0.0457 ± 0.0164 | 0.0266 ± 0.0030 |
| p3b_low_hard_hybrid | 3 | 0.0095 ± 0.0000 | 0.0260 ± 0.0018 | 0.0421 ± 0.0059 | 0.0157 ± 0.0008 | 0.0397 ± 0.0122 | 0.0573 ± 0.0176 | 0.0298 ± 0.0014 |
| p3b_low_hard_shuffled | 3 | 0.0097 ± 0.0002 | 0.0235 ± 0.0008 | 0.0372 ± 0.0051 | 0.0153 ± 0.0003 | 0.0372 ± 0.0149 | 0.0526 ± 0.0217 | 0.0259 ± 0.0002 |
| p3b_low_hard_identity | 3 | 0.0096 ± 0.0001 | 0.0290 ± 0.0030 | 0.0447 ± 0.0050 | 0.0173 ± 0.0008 | 0.0371 ± 0.0069 | 0.0526 ± 0.0096 | 0.0332 ± 0.0030 |
| p3b_low_hard_random | 3 | 0.0099 ± 0.0002 | 0.0249 ± 0.0046 | 0.0380 ± 0.0095 | 0.0159 ± 0.0015 | 0.0345 ± 0.0123 | 0.0477 ± 0.0178 | 0.0273 ± 0.0052 |

### Low data (900 training stays): MSE on the 33 variables with >= 1 parent in the hybrid graph

variables: Albumin, ALT, AST, Bilirubin, BUN, Creatinine, DiasABP, FiO2, GCS, HCO3, HCT, HR, K, Lactate, Mg, MAP, MechVent, Na, NIDiasABP, NIMAP, NISysABP, PaCO2, PaO2, pH, Platelets, RespRate, SaO2, SysABP, Temp, TroponinI, TroponinT, Urine, WBC

| config | seeds | none | random:0.5 | random:0.7 | bursty:0.5 | variable:0.3 | variable:0.5 | stride:3 |
|---|---|---|---|---|---|---|---|---|
| p3b_low_plain | 3 | 0.0088 ± 0.0003 | 0.0179 ± 0.0002 | 0.0236 ± 0.0005 | 0.0141 ± 0.0003 | 0.0226 ± 0.0038 | 0.0289 ± 0.0049 | 0.0151 ± 0.0005 |
| p3b_low_hard_llm | 3 | 0.0087 ± 0.0001 | 0.0197 ± 0.0009 | 0.0277 ± 0.0004 | 0.0151 ± 0.0004 | 0.0225 ± 0.0026 | 0.0304 ± 0.0036 | 0.0164 ± 0.0011 |
| p3b_low_hard_est | 3 | 0.0088 ± 0.0002 | 0.0186 ± 0.0010 | 0.0260 ± 0.0028 | 0.0151 ± 0.0004 | 0.0203 ± 0.0038 | 0.0272 ± 0.0057 | 0.0158 ± 0.0013 |
| p3b_low_hard_hybrid | 3 | 0.0088 ± 0.0003 | 0.0205 ± 0.0013 | 0.0288 ± 0.0029 | 0.0154 ± 0.0009 | 0.0235 ± 0.0046 | 0.0318 ± 0.0067 | 0.0179 ± 0.0013 |
| p3b_low_hard_shuffled | 3 | 0.0087 ± 0.0005 | 0.0189 ± 0.0008 | 0.0268 ± 0.0034 | 0.0146 ± 0.0004 | 0.0230 ± 0.0064 | 0.0317 ± 0.0105 | 0.0155 ± 0.0001 |
| p3b_low_hard_identity | 3 | 0.0087 ± 0.0001 | 0.0206 ± 0.0009 | 0.0284 ± 0.0016 | 0.0152 ± 0.0001 | 0.0222 ± 0.0023 | 0.0297 ± 0.0032 | 0.0182 ± 0.0008 |
| p3b_low_hard_random | 3 | 0.0087 ± 0.0003 | 0.0196 ± 0.0022 | 0.0269 ± 0.0047 | 0.0149 ± 0.0011 | 0.0216 ± 0.0056 | 0.0295 ± 0.0091 | 0.0158 ± 0.0023 |

## Full data (7,200 training stays): overall test MSE

| config | seeds | none | random:0.5 | random:0.7 | bursty:0.5 | variable:0.3 | variable:0.5 | stride:3 |
|---|---|---|---|---|---|---|---|---|
| p0_quite_itr | 3 | 0.0054 ± 0.0001 | 0.0119 ± 0.0006 | 0.0164 ± 0.0011 | 0.0079 ± 0.0002 | 0.0166 ± 0.0015 | 0.0225 ± 0.0023 | 0.0108 ± 0.0006 |
| p3phys_hard_llm | 1 | 0.0056 | 0.0128 | 0.0184 | 0.0086 | 0.0149 | 0.0214 | 0.0114 |
| p3phys_soft_llm_lam2 | 2 | 0.0055 ± 0.0002 | 0.0126 ± 0.0005 | 0.0185 ± 0.0005 | 0.0085 ± 0.0003 | 0.0213 ± 0.0028 | 0.0304 ± 0.0042 | 0.0111 ± 0.0007 |
| p3phys_learned | 1 | 0.0055 | 0.0111 | 0.0165 | 0.0076 | 0.0166 | 0.0228 | 0.0095 |

### Full data (7,200 training stays): MSE on the 33 variables with >= 1 parent in the llm graph

variables: Albumin, ALT, AST, Bilirubin, BUN, Creatinine, DiasABP, FiO2, GCS, HCO3, HCT, HR, K, Lactate, Mg, MAP, MechVent, Na, NIDiasABP, NIMAP, NISysABP, PaCO2, PaO2, pH, Platelets, RespRate, SaO2, SysABP, Temp, TroponinI, TroponinT, Urine, WBC

| config | seeds | none | random:0.5 | random:0.7 | bursty:0.5 | variable:0.3 | variable:0.5 | stride:3 |
|---|---|---|---|---|---|---|---|---|
| p0_quite_itr | 3 | 0.0054 ± 0.0001 | 0.0126 ± 0.0006 | 0.0176 ± 0.0012 | 0.0081 ± 0.0002 | 0.0177 ± 0.0017 | 0.0242 ± 0.0026 | 0.0114 ± 0.0007 |
| p3phys_hard_llm | 1 | 0.0056 | 0.0137 | 0.0198 | 0.0089 | 0.0158 | 0.0229 | 0.0121 |
| p3phys_soft_llm_lam2 | 2 | 0.0055 ± 0.0001 | 0.0133 ± 0.0005 | 0.0198 ± 0.0005 | 0.0087 ± 0.0003 | 0.0230 ± 0.0031 | 0.0329 ± 0.0047 | 0.0117 ± 0.0007 |
| p3phys_learned | 1 | 0.0055 | 0.0117 | 0.0176 | 0.0078 | 0.0177 | 0.0245 | 0.0099 |

### Full data (7,200 training stays): MSE on the 11 variables with >= 1 parent in the estimated graph

variables: DiasABP, FiO2, MAP, NIDiasABP, NIMAP, PaCO2, PaO2, RespRate, SysABP, Temp, Urine

| config | seeds | none | random:0.5 | random:0.7 | bursty:0.5 | variable:0.3 | variable:0.5 | stride:3 |
|---|---|---|---|---|---|---|---|---|
| p0_quite_itr | 3 | 0.0059 ± 0.0000 | 0.0151 ± 0.0007 | 0.0207 ± 0.0011 | 0.0091 ± 0.0003 | 0.0198 ± 0.0018 | 0.0263 ± 0.0021 | 0.0179 ± 0.0007 |
| p3phys_hard_llm | 1 | 0.0060 | 0.0138 | 0.0205 | 0.0088 | 0.0168 | 0.0243 | 0.0159 |
| p3phys_soft_llm_lam2 | 2 | 0.0060 ± 0.0001 | 0.0167 ± 0.0005 | 0.0258 ± 0.0010 | 0.0098 ± 0.0000 | 0.0316 ± 0.0088 | 0.0459 ± 0.0139 | 0.0195 ± 0.0003 |
| p3phys_learned | 1 | 0.0059 | 0.0144 | 0.0237 | 0.0091 | 0.0233 | 0.0325 | 0.0161 |

### Full data (7,200 training stays): MSE on the 33 variables with >= 1 parent in the hybrid graph

variables: Albumin, ALT, AST, Bilirubin, BUN, Creatinine, DiasABP, FiO2, GCS, HCO3, HCT, HR, K, Lactate, Mg, MAP, MechVent, Na, NIDiasABP, NIMAP, NISysABP, PaCO2, PaO2, pH, Platelets, RespRate, SaO2, SysABP, Temp, TroponinI, TroponinT, Urine, WBC

| config | seeds | none | random:0.5 | random:0.7 | bursty:0.5 | variable:0.3 | variable:0.5 | stride:3 |
|---|---|---|---|---|---|---|---|---|
| p0_quite_itr | 3 | 0.0054 ± 0.0001 | 0.0126 ± 0.0006 | 0.0176 ± 0.0012 | 0.0081 ± 0.0002 | 0.0177 ± 0.0017 | 0.0242 ± 0.0026 | 0.0114 ± 0.0007 |
| p3phys_hard_llm | 1 | 0.0056 | 0.0137 | 0.0198 | 0.0089 | 0.0158 | 0.0229 | 0.0121 |
| p3phys_soft_llm_lam2 | 2 | 0.0055 ± 0.0001 | 0.0133 ± 0.0005 | 0.0198 ± 0.0005 | 0.0087 ± 0.0003 | 0.0230 ± 0.0031 | 0.0329 ± 0.0047 | 0.0117 ± 0.0007 |
| p3phys_learned | 1 | 0.0055 | 0.0117 | 0.0176 | 0.0078 | 0.0177 | 0.0245 | 0.0099 |
