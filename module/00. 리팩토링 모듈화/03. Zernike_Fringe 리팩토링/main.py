import pandas as pd
from datetime import datetime
import logging


from zernike_analysis import zernike_analysis


if __name__ == "__main__":
    # 데이터 불러오기
    df_rawdata = pd.read_csv("RawData-1.csv")
    logging.info(f"Raw data loaded. Shape: {df_rawdata.shape}")

     # Zernike 분석 실행
    max_index = 64
    logging.info("Starting Zernike analysis")
    df_z_coeff, df_rawdata_with_predictions = zernike_analysis(df_rawdata, max_index=max_index)

    # Z계수 nm단위로 결과 저장
    df_z_coeff.to_csv("Fringe_Zernike_Coefficients.csv", index=False)
    logging.info("Zernike coefficients saved to Zernike_Coefficients.csv")


    df_rawdata_with_predictions.to_csv("Zernike_Fit.csv", index=False)
    logging.info("Zernike predictions and residuals saved to Z_FIT.csv")

