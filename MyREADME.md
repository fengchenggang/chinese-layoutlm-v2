# 文件说明
- ner_visualize.py 实体实体的可视化
- re_visualize.py 关系的可视化
- results_process_for_re.py  将对齐的实体识别结果处理成re输入格式
- pred_data_process.py 在进行实体识别前，对bbox进行 行对齐


# 调试
- 使用行对齐操作，使得模型在推理阶段f1提升1.2%


# 运行
- python run_xfun_ser.py 进行实体识别
- python run_xfun_re.py 进行关系抽取