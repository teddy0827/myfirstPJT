# 시료
## B3N049.1_VH075040_VH075030_20240530171740_11.nau
## PDS211.1_WF075040_WF075030_20240716235504_11.nau



# 20240917
### 1. 03.2 ★ X_reg + MRC_X.ipynb 
    Point MRC가 M3S계산값에 포함되는지 확인해보자. 
    -> 들어가있다.   X_REG + MRC_X 상태로 계산해줌. (당연히 K MRC는 DECORRECT해줌. )    MRC_X 에 PSM, POINT_MRC가 다 들어가있음. ( MRC_X = -PSM + Point MRC ) 
       OCM과 동일하게 한다면 ?  ADI에서는 X_REG_demrc + MRC_X 로 계산해주면 됨.   ※ OCO에서는 PSM INPUT 빼주면 안됨. 순수 X_REG로만 M3S 계산하면 됌. 

### 2. 08.4 ★ CPE RK5부호확인_240917.ipynb
    RK5에 마이너스 넣어야될거같다...
    -> CPE Regression & Fitting + TROCS Input Fitting 할때에  RK5에만 부호반대 처리해줌. 

# 20240919
### 3. 10.5 7차 기존방식.ipynb
    trocs input k값을 fitting하려고 함.  X_dx * K = Y_dx(pred) 
    3차까지는 fit값 문제없어보임. (WF7.5 HO-TROCS사용에서는 FIT값 동일함).   
    근데, hyper para사용하는 vh7.5는 (rk21~rk72) 값이 너무 큼... RK값(TROCS INPUT시트)을 보니,  RK21부터는 스케일조정이 좀 다른듯함. 확인을 좀 해봐야겠음. 

    Case 3  ' GPM_hybrid    
        unit_off = 1          'um                  canon 단위
        unit_1st = 10 ^ -6    'ppm
        unit_2nd = 10 ^ -9    'ppg/um
        unit_3rd = 10 ^ -12   'ppt/um^2
        
        unit_4th = 10 ^ -19   'nm/cm^4      ASML 단위
        unit_5th = 10 ^ -23   'nm/cm^5
        unit_6th = 10 ^ -27   'nm/cm^6
        unit_7th = 10 ^ -31   'nm/cm^7
        unit_8th = 10 ^ -35   'nm/cm^8
        unit_9th = 10 ^ -39   'nm/cm^9
        unit_nm = 10 ^ -3     'nm


        
    

