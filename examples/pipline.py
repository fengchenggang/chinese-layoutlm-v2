# -*- coding: utf-8 -*-
import base64
import copy
import numpy as np
import traceback
import cv2

from ruizhen_ocr import RuizhenAngle
from flask import Flask
from flask import request
from layoutlmft.data.utils import load_image, merge_bbox, normalize_bbox, simplify_bbox
from transformers import AutoTokenizer
from run_xfun_ser import ner_infer
from run_xfun_re import re_infer

ruizhen_ocr = RuizhenAngle()
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

entity_labels = ["HEADER", "QUESTION", "ANSWER"]
entity_label_index_map = {label: index for index, label in enumerate(entity_labels)}
index_entity_label_map = {index: label for index, label in enumerate(entity_labels)}


def gene_ner_data(bin_file):
    ocr_res = ruizhen_ocr.request_in_image_file(bin_file)
    rec_results = ocr_res['response']['data']['identify_results'][0]['details']['text2']

    if isinstance(bin_file, str):
        image, size = load_image(bin_file)
    else:
        image = bin_file
        size = (image.shape[1], image.shape[0])
    tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
    bbox_src_size = []

    for line in rec_results:
        # line_box = [line['regions'][0], line['regions'][1],
        #             line['regions'][2], line['regions'][3]]
        # line_box = [int(p) for p in line_box]

        tokenized_inputs = tokenizer(
            line['result'],
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )

        text_length = 0
        ocr_length = 0
        bbox = []
        last_box = None
        for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
            if token_id == 6:
                bbox.append(None)
                bbox_src_size.append(None)
                continue
            text_length += offset[1] - offset[0]
            tmp_box = []
            while ocr_length < text_length:
                ocr_word = line["words"].pop(0)
                ocr_length += len(
                    tokenizer._tokenizer.normalizer.normalize_str(ocr_word["result"].strip())
                )
                word_box = [ocr_word['region'][0], ocr_word['region'][1],
                            ocr_word['region'][2], ocr_word['region'][3]]
                word_box = [int(p) for p in word_box]
                tmp_box.append(simplify_bbox(word_box))
            if len(tmp_box) == 0:
                tmp_box = last_box
            bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
            bbox_src_size.append(merge_bbox(tmp_box))
            last_box = tmp_box

        bbox = [
            [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
            for i, b in enumerate(bbox)
        ]
        bbox_src_size = [
            [bbox_src_size[i + 1][0], bbox_src_size[i + 1][1], bbox_src_size[i + 1][0],
             bbox_src_size[i + 1][1]] if b is None else b for i, b in enumerate(bbox_src_size)
        ]
        label = [0] * len(bbox)
        tokenized_inputs.update({"bbox": bbox, "labels": label})
        for i in tokenized_doc:
            tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]
    assert len(bbox_src_size) == len(tokenized_doc['bbox'])
    item = {}
    for k in tokenized_doc:
        item[k] = tokenized_doc[k]
    item['image'] = image
    return [item], image, bbox_src_size


def gene_entities(results):
    entities = []
    end = -1
    for index, p in enumerate(results):
        if index < end:
            continue
        if p == 'O':
            end = index + 1
            while end < len(results) and results[end] == 'O':
                end += 1
        else:
            tag = p.split('-')[-1]
            end = index + 1
            while end < len(results) and results[end] == 'I-' + tag:
                end += 1
            entities.append({'start': index, 'end': end, 'label': entity_label_index_map[tag]})

    def merge(ents):
        new_ents = {key: [] for key in ents[0].keys()}
        for ent in ents:
            for k, v in ent.items():
                new_ents[k] += [v]
        return new_ents

    return merge(entities)


def gene_re_data(dataset, ner_res):
    assert len(dataset) == len(ner_res) == 1

    dataset[0]['entities'] = gene_entities(ner_res[0])
    dataset[0]['relations'] = {'head': [], 'tail': [], 'start_index': [], 'end_index': []}
    return dataset


def get_ent_bbox(ent_bbox):
    x1, y1, x2, y2 = [], [], [], []
    for bb in ent_bbox:
        x1.append(bb[0])
        y1.append(bb[1])
        x2.append(bb[2])
        y2.append(bb[3])
    x1 = min(x1)
    y1 = min(y1)
    x2 = max(x2)
    y2 = max(y2)
    return (x1, y1), (x2, y2), (int((x1 + x2) / 2), int((y1 + y2) / 2))


def draw_rec(img, results, bbox_src_size):
    results = results[0]

    for index, p in enumerate(results):
        head = p['head']
        head_type = index_entity_label_map[p['head_type']]
        tail = p['tail']
        tail_type = index_entity_label_map[p['tail_type']]

        head_bbox = bbox_src_size[head[0]:head[1]]
        tail_bbox = bbox_src_size[tail[0]:tail[1]]

        head_p1, head_p2, head_c = get_ent_bbox(head_bbox)
        tail_p1, tail_p2, tail_c = get_ent_bbox(tail_bbox)

        if head_type == 'QUESTION':
            cv2.rectangle(img, head_p1, head_p2, (0, 255, 0), 1)
        elif head_type == 'ANSWER':
            cv2.rectangle(img, head_p1, head_p2, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, head_p1, head_p2, (255, 0, 0), 1)

        if tail_type == 'QUESTION':
            cv2.rectangle(img, tail_p1, tail_p2, (0, 255, 0), 1)
        elif head_type == 'ANSWER':
            cv2.rectangle(img, tail_p1, tail_p2, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, tail_p1, tail_p2, (255, 0, 0), 1)

        cv2.line(img, head_c, tail_c, (0, 0, 255), 1)

    return img


def kv_extract(bin_file):
    ner_test_dataset, image, bbox_src_size = gene_ner_data(bin_file)
    ner_res = ner_infer(copy.deepcopy(ner_test_dataset))
    re_test_dataset = gene_re_data(ner_test_dataset, ner_res)
    re_res = re_infer(re_test_dataset)
    return draw_rec(image, re_res, bbox_src_size)


@app.route("/kv_extract", methods=['POST', 'GET'])
def extract():
    try:
        img = base64.b64decode(str(request.form['image']))
        image_data = np.fromstring(img, np.uint8)
        image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        image = kv_extract(image_data)

        image = cv2.imencode('.jpg', image)[1]
        base64_data = str(base64.b64encode(image))[2:-1]
        return {'state': 'succeed', 'img': base64_data}
    except:
        traceback.print_exc()
        return {"state": 'failed'}


if __name__ == "__main__":
    app.run(host='localhost', port=10003, debug=True)

    # 测试
    # gene_ner_data(r'/work/Codes/layoutlmft/examples/XFUND-DATA-Gartner/zh.val/zh_val_0.jpg')

    # kv_extract(r'/work/Codes/layoutlmft/examples/XFUND-DATA-Gartner/zh.val/zh_val_0.jpg')
