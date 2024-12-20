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


        
    
### 잔여 작업 할것
1. ADI M3S계산을 OCM과 맞춰주려고 함.   X_REG_demrc + MRC_X 계산해서 신규컬럼 추가.
2. WAFERMAP 배열 수정. MRC포함된 FITTING MAP은 필요없어보임.   Raw의 기준을  X_REG_demrc + MRC_X  으로 변경하고. (ocm의 E1과 같음)   이걸 FITTING한것으로 바꾸자. 
3. NAU여러개로 TREND 차트 구성 좀 생각해보자.
4. 



```

import pandas as pd
import numpy as np
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_log.log"),
        logging.StreamHandler()
    ]
)

def construct_X_matrix(x, y, rx, ry, matrix_type='dx'):
    """
    X 행렬을 구성하는 함수.

    Parameters:
        x (numpy.ndarray): X 좌표
        y (numpy.ndarray): Y 좌표
        rx (numpy.ndarray): Field X 좌표
        ry (numpy.ndarray): Field Y 좌표
        matrix_type (str): 'dx' 또는 'dy'로 구분하여 dx/dy에 맞는 행렬 생성
    
    Returns:
        numpy.ndarray: 구성된 X 행렬
    """
    if matrix_type == 'dx':  # WK 1023/1023, RK 1023/511
        terms = [
            np.ones(len(x)), 
            x / 1e6, -y / 1e6, 
            (x ** 2) / 1e12, (x * y) / 1e12, (y ** 2) / 1e12, 
            (x ** 3) / 1e15, (x ** 2 * y) / 1e15, (x * y ** 2) / 1e15, (y ** 3) / 1e15,
            rx / 1e6, -ry / 1e6, 
            (rx ** 2) / 1e9, (rx * ry) / 1e9, (ry ** 2) / 1e9,
            (rx ** 3) / 1e12, (rx ** 2 * ry) / 1e12, (rx * ry ** 2) / 1e12, (ry ** 3) / 1e12
        ]
    elif matrix_type == 'dy':
        terms = [
            np.ones(len(y)), 
            y / 1e6, x / 1e6, 
            (y ** 2) / 1e12, (y * x) / 1e12, (x ** 2) / 1e12, 
            (y ** 3) / 1e15, (y ** 2 * x) / 1e15, (y * x ** 2) / 1e15, (x ** 3) / 1e15,
            ry / 1e6, rx / 1e6, (ry ** 2) / 1e9, (ry * rx) / 1e9, (rx ** 2) / 1e9,
            (ry ** 3) / 1e12, (ry ** 2 * rx) / 1e12, (ry * rx ** 2) / 1e12
        ]
    else:
        raise ValueError("matrix_type은 'dx' 또는 'dy'만 가능합니다.")
    
    return np.vstack(terms).T


def prepare_coordinates(group):
    """
    좌표 및 독립변수를 계산하는 공통 함수.

    Parameters:
        group (pandas.DataFrame): UNIQUE_ID 그룹 데이터
    
    Returns:
        tuple: (x, y, rx, ry) 계산된 좌표
    """
    die_x = group['DieX']
    die_y = group['DieY']
    step_pitch_x = group['STEP_PITCH_X']
    step_pitch_y = group['STEP_PITCH_Y']
    map_shift_x = group['MAP_SHIFT_X']
    map_shift_y = group['MAP_SHIFT_Y']
    coordinate_x = group['coordinate_X']
    coordinate_y = group['coordinate_Y']

    # 좌표 계산
    x = die_x * step_pitch_x + map_shift_x
    y = die_y * step_pitch_y + map_shift_y
    rx = coordinate_x
    ry = coordinate_y
    
    return x, y, rx, ry


def create_matrices(x, y, rx, ry):
    """
    X_dx와 X_dy 행렬을 생성하는 공통 함수.

    Parameters:
        x, y, rx, ry (numpy.ndarray): 좌표 데이터
    
    Returns:
        tuple: (X_dx, X_dy) 행렬
    """
    X_dx = construct_X_matrix(x, y, rx, ry, matrix_type='dx')
    X_dy = construct_X_matrix(x, y, rx, ry, matrix_type='dy')
    return X_dx, X_dy


def multi_lot_analysis(df_rawdata):
    """
    UNIQUE_ID별로 회귀 및 잔차 계산을 통합 수행하는 함수.
    df_rawdata에 잔차 결과를 직접 추가합니다.

    Parameters:
        df_rawdata (pandas.DataFrame): 입력 데이터
    
    Returns:
        pandas.DataFrame: 회귀 계수 결과 데이터프레임
    """
    grouped = df_rawdata.groupby('UNIQUE_ID')
    coeff_results = []  # 회귀 계수 저장
    
    # 예측값과 잔차를 저장할 열 초기화
    df_rawdata['pred_x'] = np.nan
    df_rawdata['pred_y'] = np.nan
    df_rawdata['residual_x'] = np.nan
    df_rawdata['residual_y'] = np.nan




    for unique_id, group in grouped:
        logging.info(f"Processing UNIQUE_ID: {unique_id}")

        # 좌표 및 독립변수 준비
        x, y, rx, ry = prepare_coordinates(group)
        X_dx, X_dy = create_matrices(x, y, rx, ry)

        # 종속변수
        Y_dx = group['X_reg']
        Y_dy = group['Y_reg']

        # 회귀 계수 계산
        coeff_dx = np.linalg.lstsq(X_dx, Y_dx, rcond=None)[0]
        coeff_dy = np.linalg.lstsq(X_dy, Y_dy, rcond=None)[0]

        # 회귀 계수 결과 저장
        coeff_results.append(pd.DataFrame({
            'UNIQUE_ID': [unique_id],
            'WK1': [coeff_dx[0]],
            'WK2': [coeff_dy[0]],
            'WK3': [coeff_dx[1]],
            'WK4': [coeff_dy[1]],
            'WK5': [coeff_dx[2]],
            'WK6': [coeff_dy[2]],
            'WK7': [coeff_dx[3]],
            'WK8': [coeff_dy[3]],
            'WK9': [coeff_dx[4]],
            'WK10': [coeff_dy[4]],
            'WK11': [coeff_dx[5]],
            'WK12': [coeff_dy[5]],
            'WK13': [coeff_dx[6]],
            'WK14': [coeff_dy[6]],
            'WK15': [coeff_dx[7]],
            'WK16': [coeff_dy[7]],
            'WK17': [coeff_dx[8]],
            'WK18': [coeff_dy[8]],
            'WK19': [coeff_dx[9]],
            'WK20': [coeff_dy[9]],
            'RK1': [0],
            'RK2': [0],
            'RK3': [coeff_dx[10]],
            'RK4': [coeff_dy[10]],
            'RK5': [coeff_dx[11]],
            'RK6': [coeff_dy[11]],
            'RK7': [coeff_dx[12]],
            'RK8': [coeff_dy[12]],
            'RK9': [coeff_dx[13]],
            'RK10': [coeff_dy[13]],
            'RK11': [coeff_dx[14]],
            'RK12': [coeff_dy[14]],
            'RK13': [coeff_dx[15]],
            'RK14': [coeff_dy[15]],
            'RK15': [coeff_dx[16]],
            'RK16': [coeff_dy[16]],
            'RK17': [coeff_dx[17]],
            'RK18': [coeff_dy[17]],
            'RK19': [coeff_dx[18]],
            'RK20': [0]
        }))


        # 예측값 계산
        pred_x = X_dx.dot(coeff_dx)
        pred_y = X_dy.dot(coeff_dy)

        # 잔차 계산
        residual_x = group['X_reg'] - pred_x
        residual_y = group['Y_reg'] - pred_y


        # 기존 df_rawdata에 예측값과 잔차 추가
        df_rawdata.loc[group.index, 'pred_x'] = pred_x
        df_rawdata.loc[group.index, 'pred_y'] = pred_y
        df_rawdata.loc[group.index, 'residual_x'] = residual_x
        df_rawdata.loc[group.index, 'residual_y'] = residual_y


    # 결과 병합
    df_coeff = pd.concat(coeff_results, ignore_index=True)
  
    return df_coeff, df_rawdata






def compute_fringe_zernike_matrix(r, theta, max_index):
    """
    ASML Fringe Zernike 다항식 기저 생성 (Z36까지)
    :param r: 방사 좌표계의 반지름 값 (0 ≤ r ≤ 1로 스케일링 필요)
    :param theta: 방사 좌표계의 각도 값 (-π ≤ θ ≤ π)
    :param max_index: 최대 인덱스 (Z1부터 Z36까지)
    :return: Zernike 기저 행렬
    """
    Z = [np.ones_like(r)]  # Z1: 상수항
    if max_index >= 2:
        Z.append(r * np.cos(theta))  # Z2
    if max_index >= 3:
        Z.append(r * np.sin(theta))  # Z3
    if max_index >= 4:
        Z.append(2 * r**2 - 1)  # Z4
    if max_index >= 5:
        Z.append(r**2 * np.cos(2 * theta))  # Z5
    if max_index >= 6:
        Z.append(r**2 * np.sin(2 * theta))  # Z6
    if max_index >= 7:
        Z.append((3 * r**3 - 2 * r) * np.cos(theta))  # Z7
    if max_index >= 8:
        Z.append((3 * r**3 - 2 * r) * np.sin(theta))  # Z8
    if max_index >= 9:
        Z.append(6 * r**4 - 6 * r**2 + 1)  # Z9
    if max_index >= 10:
        Z.append(r**3 * np.cos(3 * theta))  # Z10
    if max_index >= 11:
        Z.append(r**3 * np.sin(3 * theta))  # Z11
    if max_index >= 12:
        Z.append((4 * r**4 - 3 * r**2) * np.cos(2 * theta))  # Z12
    if max_index >= 13:
        Z.append((4 * r**4 - 3 * r**2) * np.sin(2 * theta))  # Z13
    if max_index >= 14:
        Z.append((10 * r**5 - 12 * r**3 + 3 * r) * np.cos(theta))  # Z14
    if max_index >= 15:
        Z.append((10 * r**5 - 12 * r**3 + 3 * r) * np.sin(theta))  # Z15
    if max_index >= 16:
        Z.append(20 * r**6 - 30 * r**4 + 12 * r**2 - 1)  # Z16
    if max_index >= 17:
        Z.append(r**4 * np.cos(4 * theta))  # Z17
    if max_index >= 18:
        Z.append(r**4 * np.sin(4 * theta))  # Z18
    if max_index >= 19:
        Z.append((5 * r**5 - 4 * r**3) * np.cos(3 * theta))  # Z19
    if max_index >= 20:
        Z.append((5 * r**5 - 4 * r**3) * np.sin(3 * theta))  # Z20
    if max_index >= 21:
        Z.append((15 * r**6 - 20 * r**4 + 6 * r**2) * np.cos(2 * theta))  # Z21
    if max_index >= 22:
        Z.append((15 * r**6 - 20 * r**4 + 6 * r**2) * np.sin(2 * theta))  # Z22
    if max_index >= 23:
        Z.append((35 * r**7 - 60 * r**5 + 30 * r**3 - 4 * r) * np.cos(theta))  # Z23
    if max_index >= 24:
        Z.append((35 * r**7 - 60 * r**5 + 30 * r**3 - 4 * r) * np.sin(theta))  # Z24
    if max_index >= 25:
        Z.append(70 * r**8 - 140 * r**6 + 90 * r**4 - 20 * r**2 + 1)  # Z25
    if max_index >= 26:
        Z.append(r**5 * np.cos(5 * theta))  # Z26
    if max_index >= 27:
        Z.append(r**5 * np.sin(5 * theta))  # Z27
    if max_index >= 28:
        Z.append((6 * r**6 - 5 * r**4) * np.cos(4 * theta))  # Z28
    if max_index >= 29:
        Z.append((6 * r**6 - 5 * r**4) * np.sin(4 * theta))  # Z29
    if max_index >= 30:
        Z.append((21 * r**7 - 30 * r**5 + 10 * r**3) * np.cos(3 * theta))  # Z30
    if max_index >= 31:
        Z.append((21 * r**7 - 30 * r**5 + 10 * r**3) * np.sin(3 * theta))  # Z31
    if max_index >= 32:
        Z.append((56 * r**8 - 105 * r**6 + 60 * r**4 - 10 * r**2) * np.cos(2 * theta))  # Z32
    if max_index >= 33:
        Z.append((56 * r**8 - 105 * r**6 + 60 * r**4 - 10 * r**2) * np.sin(2 * theta))  # Z33
    if max_index >= 34:
        Z.append((126 * r**9 - 280 * r**7 + 210 * r**5 - 60 * r**3 + 5 * r) * np.cos(theta))  # Z34
    if max_index >= 35:
        Z.append((126 * r**9 - 280 * r**7 + 210 * r**5 - 60 * r**3 + 5 * r) * np.sin(theta))  # Z35
    if max_index >= 36:
        Z.append(252 * r**10 - 630 * r**8 + 560 * r**6 - 210 * r**4 + 30 * r**2 - 1)  # Z36
    return np.array(Z).T  # 각 다항식이 열로 구성된 행렬




# 좌표 변환 및 Zernike 기저 생성 함수
def prepare_zernike_coordinates(group, max_index):
    """
    Zernike 좌표 및 기저 행렬 생성
    :param group: UNIQUE_ID별 그룹 데이터
    :param max_order: 최대 차수
    :return: Zernike 기저 행렬
    """
    die_x = group['DieX']
    die_y = group['DieY']
    step_pitch_x = group['STEP_PITCH_X']
    step_pitch_y = group['STEP_PITCH_Y']
    map_shift_x = group['MAP_SHIFT_X']
    map_shift_y = group['MAP_SHIFT_Y']
    coordinate_x = group['coordinate_X']
    coordinate_y = group['coordinate_Y']

    # 방사 좌표계 변환
    wf_x = die_x * step_pitch_x + map_shift_x + coordinate_x
    wf_y = die_y * step_pitch_y + map_shift_y + coordinate_y
    r = np.sqrt(wf_x**2 + wf_y**2) / 1e6  # 거리 스케일링
    theta = np.arctan2(wf_y, wf_x)

    # Zernike 기저 행렬 생성
    return compute_fringe_zernike_matrix(r, theta, max_index=max_index)


# Zernike 분석 함수
def zernike_analysis(df_rawdata, max_index):
    """
    Zernike 회귀분석 및 잔차 계산
    :param df_rawdata: 입력 데이터
    :param max_order: Zernike 다항식 최대 차수
    :return: (df_z_coeff, df_rawdata_with_predictions)
    """
    grouped = df_rawdata.groupby('UNIQUE_ID')
    coeff_results = []

    # 원본 데이터프레임에 예측값 및 잔차 열 추가
    df_rawdata['Z_pred_x'] = np.nan
    df_rawdata['Z_pred_y'] = np.nan
    df_rawdata['Z_residual_x'] = np.nan
    df_rawdata['Z_residual_y'] = np.nan

    for unique_id, group in grouped:
        logging.info(f"Processing UNIQUE_ID: {unique_id}")

        # Zernike 기저 생성
        Z = prepare_zernike_coordinates(group, max_index=max_index)

        # 종속변수
        Y_dx = group['X_reg']
        Y_dy = group['Y_reg']

        # 회귀 계수 계산
        coeff_dx = np.linalg.lstsq(Z, Y_dx, rcond=None)[0]
        coeff_dy = np.linalg.lstsq(Z, Y_dy, rcond=None)[0]

        # 회귀 계수 저장
        coeff_result = {'UNIQUE_ID': unique_id}
        coeff_result.update({f'Z{i+1}_dx': coeff for i, coeff in enumerate(coeff_dx)})
        coeff_result.update({f'Z{i+1}_dy': coeff for i, coeff in enumerate(coeff_dy)})
        coeff_results.append(coeff_result)

        # 예측값 계산
        pred_x = Z @ coeff_dx
        pred_y = Z @ coeff_dy

        # 잔차 계산
        residual_x = Y_dx - pred_x
        residual_y = Y_dy - pred_y

        # 원본 데이터프레임에 추가
        df_rawdata.loc[group.index, 'Z_pred_x'] = pred_x
        df_rawdata.loc[group.index, 'Z_pred_y'] = pred_y
        df_rawdata.loc[group.index, 'Z_residual_x'] = residual_x
        df_rawdata.loc[group.index, 'Z_residual_y'] = residual_y

    # 회귀 계수 결과 데이터프레임 생성
    df_z_coeff = pd.DataFrame(coeff_results)
    return df_z_coeff, df_rawdata





if __name__ == "__main__":
    # 데이터 불러오기
    df_rawdata = pd.read_csv("RawData-1.csv")
    logging.info(f"Raw data loaded. Shape: {df_rawdata.shape}")

    # 통합 분석 실행
    logging.info("Starting multi-lot analysis")
    df_coeff, df_residual = multi_lot_analysis(df_rawdata)

    # 결과 저장
    df_coeff.to_csv("OSR_K_test.csv", index=False)
    logging.info("Regression coefficients saved to OSR_K_test.csv")


    # Zernike 분석 실행
    max_index = 36
    logging.info("Starting Zernike analysis")
    df_z_coeff, df_rawdata_with_predictions = zernike_analysis(df_rawdata, max_index=max_index)

    # 결과 저장
    df_z_coeff.to_csv("Fringe_Zernike_Coefficients.csv", index=False)
    logging.info("Zernike coefficients saved to Zernike_Coefficients.csv")


    df_rawdata_with_predictions.to_csv("통합(C+FZ)_FIT.csv", index=False)
    logging.info("Zernike predictions and residuals saved to Z_FIT.csv")

    

```


