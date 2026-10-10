# 밀라노 통신 트래픽 예측 실험

시간 정보를 추가하면 통신 트래픽 예측이 더 정확해지는지 확인한 실험입니다. 밀라노의 실제 통신 데이터를 이용해, 과거 트래픽만 입력한 GRU와 시간대·요일·주말 정보를 함께 입력한 GRU를 비교했습니다.

## 실험 목적

트래픽은 시간대와 요일에 따라 달라질 수 있습니다. 과거 트래픽만으로도 이러한 패턴을 학습할 수 있지만, 시간 정보를 직접 제공하면 예측에 도움이 되는지 확인하고자 했습니다.

데이터는 2013년 11월 1일부터 28일까지의 Milano Square ID 1을 사용했습니다. 같은 시각의 국가 코드별 Internet activity를 합산해 10분 간격의 시계열로 구성했습니다. 총 4,032개 시점이며, 집계 후 결측 구간은 없었습니다.

## 실험 방법

두 모델 모두 과거 24시간을 입력하고, 1·3·6·12·24시간 후의 활동값 한 개를 예측했습니다. 예측 구간마다 별도의 모델을 학습하는 direct forecasting 방식입니다.

- 기본 GRU에는 과거 Internet activity만 입력했습니다.
- 시간 정보 GRU에는 동일한 활동값과 시간대 sin/cos, 요일 sin/cos, 주말 여부를 입력했습니다.
- Persistence는 마지막 관측값이 미래에도 유지된다고 가정한 비교 기준입니다.

데이터는 시간순으로 학습 70%, 검증 15%, 시험 15%로 나누었습니다. 정규화는 학습 데이터의 평균과 표준편차만 사용했습니다. 두 GRU에 같은 학습 설정과 seed 42·43·44를 적용했으며, 검증 데이터로 학습 종료 시점을 결정했습니다.

같은 예측 구간에서는 두 모델과 Persistence를 동일한 시험 시점에서 평가했습니다. 시험 기간에도 관측이 끝난 값은 이후 예측의 입력으로 사용하는 rolling-origin 평가입니다.

## 실험 결과

아래 값은 seed 3회의 평균 RMSE입니다. 감소율은 기본 GRU의 평균 RMSE를 기준으로 계산했습니다.

|예측 시점|기본 GRU|시간 정보 GRU|RMSE 감소율|
|---|---:|---:|---:|
|1시간 후|2.1433|1.9333|9.80%|
|3시간 후|2.0982|1.8447|12.08%|
|6시간 후|2.0988|1.8146|13.54%|
|12시간 후|2.1155|1.9184|9.32%|
|24시간 후|2.2020|2.0071|8.85%|

모든 예측 구간에서 시간 정보 GRU의 평균 RMSE가 낮았습니다. 가장 큰 차이는 6시간 후 예측에서 나타났으며, RMSE가 13.54% 감소했습니다. 시간 정보 GRU는 모든 예측 구간에서 Persistence보다도 낮은 RMSE를 기록했습니다.

이번 데이터에서는 시간 정보를 추가하는 것이 예측에 도움이 되었습니다. 다만 단일 지역의 28일 데이터를 사용한 결과이므로, 다른 지역이나 기간에서도 같은 개선이 나타나는지는 추가 검증이 필요합니다.

![예측 구간별 성능 비교](outputs/research_nov01_28/horizon_performance.png)

## 발표 자료와 상세 결과

- [발표 PPT](deliverables/Milano_Traffic_Portfolio.pptx)
- [데이터 검증과 전체 실험 보고서](outputs/research_nov01_28/REPORT.md)
- [seed별 R²·RMSE·MAE](outputs/research_nov01_28/metrics.csv)
- [평균과 표준편차](outputs/research_nov01_28/performance_formatted.csv)
- [기본 GRU 대비 변화](outputs/research_nov01_28/paired_differences.csv)
- [Persistence 비교](outputs/research_nov01_28/persistence_comparison.csv)

## 실행 방법

이 폴더에서 Python 환경을 준비한 뒤 아래 명령으로 실행할 수 있습니다. 학습과 평가에 사용한 집계 시계열이 포함되어 있습니다.

```shell
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests
python -m milano.experiment --series outputs/nov01_28/series.csv --config configs/experiment.json --out outputs/rerun
```

원본 파일부터 집계하려면 [Harvard Dataverse](https://doi.org/10.7910/DVN/EGZHFV)에서 2013년 11월 1~28일 파일을 내려받아 `raw` 폴더에 넣습니다. 해당 폴더에는 이번 실험 기간의 파일만 두어야 합니다.

```shell
python -m milano.preprocess --input "raw/sms-call-internet-mi-2013-11-*.txt" --square 1 --out outputs/rerun_data
```

Internet activity는 데이터셋에서 제공하는 통신활동 지표입니다. MB나 Mbps를 의미하지 않습니다. 자세한 정의는 [데이터 설명 논문](https://www.nature.com/articles/sdata201555)에서 확인할 수 있습니다.

## 실험의 한계

시험 기간은 약 4.2일이며, 예측 구간별 시험 시점 수에도 차이가 있습니다. 입력 창이 겹치고 시계열에는 자기상관이 있으므로 시험 표본을 독립적인 관측으로 볼 수 없습니다. seed 3회의 표준편차는 학습 초기화에 따른 변동을 보여주는 값이며 신뢰구간은 아닙니다.

두 GRU에 같은 학습 설정을 적용했지만, 입력 차원에 따라 파라미터 수는 다릅니다. 시간대·요일·주말 정보 중 어느 요소가 개선에 기여했는지는 따로 평가하지 않았습니다. 원 논문의 모든 모델과 지역을 재현한 실험은 아닙니다.

구현에는 Python과 PyTorch를 사용했으며, 코드와 자료 정리에 AI 보조를 활용했습니다. 실행·검증 과정의 수정 내역은 [변경 기록](EXPERIMENT_CHANGES.md)에 정리했습니다.
