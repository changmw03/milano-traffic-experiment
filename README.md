# 밀라노 통신 트래픽 예측 실험

밀라노의 실제 통신 데이터를 이용해 시간 정보가 GRU의 예측 성능에 미치는 영향을 확인한 실험입니다.

2013년 11월 1~28일 Square ID 1의 데이터를 사용했습니다. 과거 트래픽만 입력한 기본 GRU와 시간대·요일·주말 정보를 추가한 GRU를 비교한 결과, 예측 구간에 따라 평균 RMSE가 8.85~13.54% 감소했습니다.

- [실험 소개와 실행 방법](milano-traffic-forecasting/README.md)
- [발표 PPT](milano-traffic-forecasting/deliverables/Milano_Traffic_Portfolio.pptx)
- [전체 실험 보고서](milano-traffic-forecasting/outputs/research_nov01_28/REPORT.md)
- [seed별 실험 결과](milano-traffic-forecasting/outputs/research_nov01_28/metrics.csv)

단일 지역의 28일 데이터를 사용한 결과입니다. 다른 지역이나 기간에 대한 일반화는 검증하지 않았습니다.
