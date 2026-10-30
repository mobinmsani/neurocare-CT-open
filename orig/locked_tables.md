# Values reported by the original pipeline

Excerpts of the locked result tables of the original pipeline, used by manuscript/build_si.py to check the values in the
reproduction table of the Supplementary Information that are not in out/verify_original_comparison.txt. Table rows are
reproduced as they were written by the pipeline.

## Seg-CQ500 and BHSD segmentation (locked v1 analysis)

  **Cohort**                    **Method**        **Scans**   **Mean   **Median   **Mean   **Precision**   **Recall**
                                                              Dice**   Dice**     IoU**                    
  ----------------------------- ----------------- ----------- -------- ---------- -------- --------------- ------------
  Seg-CQ500                     blast_model1      43          0.531    0.635      0.428    0.743           0.472

  Seg-CQ500                     blast_min100mm3   43          0.524    0.644      0.428    0.794           0.459

  Seg-CQ500                     blast_hu45_r2     43          0.534    0.631      0.419    0.590           0.588

  Seg-CQ500                     lightunet25d      43          0.465    0.498      0.358    0.457           0.591

  BHSD-provider-test            lightunet25d      96          0.436    0.434      0.328    0.512           0.443

## CQ500 external detection with v2 by reader-majority finding (locked CQ500 protocol; n = 489)

  ------------------------------------------------------------------------------------
  **Finding**    **Positives**   **N**   **AUC (95% CI)**   **Sensitivity at ≥1.0 mL
                                                            (95% CI)**
  -------------- --------------- ------- ------------------ --------------------------
  ICH            205             489     0.806              168/205 = 0.820
                                         (0.769--0.844)     (0.761--0.866)

  IPH            134             489     0.784              112/134 = 0.836
                                         (0.737--0.830)     (0.764--0.889)

  IVH            28              489     0.960              28/28 = 1.000
                                         (0.938--0.979)     (0.879--1.000)

  SDH            53              489     0.683              44/53 = 0.830
                                         (0.610--0.752)     (0.708--0.908)

  EDH            13              489     0.805              13/13 = 1.000
                                         (0.712--0.893)     (0.772--1.000)

  SAH            60              489     0.808              54/60 = 0.900
                                         (0.742--0.868)     (0.799--0.953)

  ChronicBleed   19              489     0.492              12/19 = 0.632
                                         (0.380--0.611)     (0.410--0.809)
  ------------------------------------------------------------------------------------

AUC uses total predicted volume as the score. Specificity at 1.0 mL was
0.595 (0.537--0.651); PPV 0.594; NPV 0.820.

## CQ500 midline shift and mass effect (locked CQ500 protocol)

  ----------------------------------------------------------------------------------
  **Reader       **Score**               **Positives**   **N**   **AUC (95% CI)**
  label**                                                        
  -------------- ----------------------- --------------- ------- -------------------
  MidlineShift   CSF-centroid offset,    62              452     0.894
                 max \|mm\|                                      (0.855--0.931)

  MidlineShift   Predicted hemorrhage    65              489     0.852
                 volume                                          (0.798--0.904)

  MassEffect     CSF-centroid offset,    118             452     0.815
                 max \|mm\|                                      (0.768--0.859)

  MassEffect     Predicted hemorrhage    127             489     0.851
                 volume                                          (0.811--0.888)
  ----------------------------------------------------------------------------------

## BHSD test set: subtype segmentation and presence with v3 and v3b (n = 96)

  ------------------------------------------------------------------------------
  **Subtype**   **Scans     **v3b    **v3     **v3b presence AUC   **v3b volume
                with        Dice**   Dice**   (95% CI)**           MAE (mL)**
                subtype**                                          
  ------------- ----------- -------- -------- -------------------- -------------
  EDH           13          0.078    0.000    0.652 (0.476--0.834) 2.9

  IPH           63          0.405    0.391    0.785 (0.684--0.876) 10.6

  IVH           54          0.287    0.235    0.746 (0.642--0.837) 2.8

  SAH           51          0.139    0.000    0.785 (0.689--0.873) 5.9

  SDH           35          0.199    0.000    0.704 (0.590--0.809) 6.0
  ------------------------------------------------------------------------------

Dice is averaged over scans containing the subtype. Version 3b is post
hoc (defined after version 3 test results). AUC uses predicted class
volume; bootstrap CIs.

## CQ500 external subtype detection with the frozen v3b model (n = 489)

  --------------------------------------------------------------------------------------------------------
  **Finding**   **Positives**   **AUC (95% CI)** **AP**   **Sensitivity   **Specificity   **AUC excl.
                                                          ≥1 mL**         ≥1 mL**         Seg-CQ500 (n =
                                                                                          446)**
  ------------- --------------- ---------------- -------- --------------- --------------- ----------------
  ANY           205             0.818            0.778    190/205 (0.927) 91/284 (0.320)  0.801
                                (0.778--0.856)                                            (0.756--0.843)

  EDH           13              0.770            0.172    4/13 (0.308)    461/476 (0.968) 0.875
                                (0.603--0.918)                                            (0.673--0.982)

  IPH           134             0.794            0.688    76/134 (0.567)  331/355 (0.932) 0.763
                                (0.746--0.840)                                            (0.706--0.817)

  IVH           28              0.948            0.630    25/28 (0.893)   423/461 (0.918) 0.965
                                (0.904--0.982)                                            (0.913--0.992)

  SAH           60              0.834            0.599    51/60 (0.850)   212/429 (0.494) 0.864
                                (0.765--0.900)                                            (0.797--0.923)

  SDH           53              0.820            0.483    45/53 (0.849)   282/436 (0.647) 0.815
                                (0.750--0.882)                                            (0.733--0.883)
  --------------------------------------------------------------------------------------------------------

Reference is the three-reader majority. Score is predicted class volume
(ANY = union). The 1-mL threshold was not tuned. AP, average precision.
