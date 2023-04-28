# GPT3訓練資料

## 模型版本對應表

online: 目前在discord的chatbot版本

| Model ID                               | 使用資料                                                                                                                   |
|----------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| davinci:ft-sysjust-2022-12-27-08-12-39 | 1. training_data_xq討論區_隨機抓.json <br/> 2. training_data_xq預設腳本_隨機抓.json <br/> 3. training_data_D大_excel_2022-12-28.json |
| davinci:ft-sysjust-2022-12-28-06-10-06 | 同上，但新增training_data_multilogits.json                                                                                   |
| (online)davinci:ft-sysjust-2022-12-28-08-39-37 | 同上，但新增training_data_marco_2022-12-28.json                                                                              |


## 資料標註

參考連結: https://beta.openai.com/docs/guides/fine-tuning/preparing-your-dataset

針對`prompt`:
1. 結尾需加上`\n\n###\n\n`

針對`completion`:
1. 開頭需空白
2. 結尾需加上`###`