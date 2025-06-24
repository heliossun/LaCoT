import os
from datasets import load_dataset
from tqdm import tqdm
import json
import base64

subfolder= ['CLEVR-Math(MathV360K)', 'FigureQA(MathV360K)', 'GEOS(MathV360K)', 'GeoQA+(MathV360K)', 'Geometry3K(MathV360K)', 'IconQA(MathV360K)', 'MapQA(MathV360K)', 'PMC-VQA(MathV360K)', 'Super-CLEVR(MathV360K)', 'TabMWP(MathV360K)', 'UniGeo(MathV360K)', 'VizWiz(MathV360K)', 'ai2d(cauldron,llava_format)', 'ai2d(gpt4v)', 'ai2d(internvl)', 'allava_instruct_laion4v', 'allava_instruct_vflan4v', 'aokvqa(cauldron,llava_format)', 'chart2text(cauldron)', 'chartqa(cauldron,llava_format)', 'chrome_writting', 'clevr(cauldron,llava_format)', 'diagram_image_to_text(cauldron)', 'dvqa(cauldron,llava_format)', 'figureqa(cauldron,llava_format)', 'geo170k(align)', 'geo170k(qa)', 'geo3k', 'geomverse(cauldron)', 'hateful_memes(cauldron,llava_format)', 'hitab(cauldron,llava_format)', 'hme100k', 'iam(cauldron)', 'iconqa(cauldron,llava_format)', 'iiit5k', 'image_textualization(filtered)', 'infographic(gpt4v)', 'infographic_vqa', 'infographic_vqa_llava_format', 'intergps(cauldron,llava_format)', 'k12_printing', 'llavar_gpt4_20k', 'lrv_chart', 'lrv_normal(filtered)', 'magpie_pro(l3_80b_mt)', 'magpie_pro(l3_80b_st)', 'magpie_pro(qwen2_72b_st)', 'mapqa(cauldron,llava_format)', 'mathqa', 'mavis_math_metagen', 'mavis_math_rule_geo', 'multihiertt(cauldron)', 'orand_car_a', 'raven(cauldron)', 'rendered_text(cauldron)', 'robut_sqa(cauldron)', 'robut_wikisql(cauldron)', 'robut_wtq(cauldron,llava_format)', 'scienceqa(cauldron,llava_format)', 'scienceqa(nona_context)', 'screen2words(cauldron)', 'sharegpt4o', 'sharegpt4v(coco)', 'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)', 'sroie', 'st_vqa(cauldron,llava_format)', 'tabmwp(cauldron)', 'tallyqa(cauldron,llava_format)', 'textcaps', 'textocr(gpt4v)', 'tqa(cauldron,llava_format)', 'ureader_cap', 'ureader_ie', 'vision_flan(filtered)', 'vistext(cauldron)', 'visual7w(cauldron,llava_format)', 'visualmrc(cauldron)', 'vqarad(cauldron,llava_format)', 'vsr(cauldron,llava_format)', 'websight(cauldron)']

image_folder = "/mnt/disks/new-disk/data/r1ov-image"
#json_file = "/home/hhua/guohao/data/r1_onevision.json"
converted_data = []

data = load_dataset("Fancy-MLLM/R1-Onevision",split="train")

root_path = os.path.join(image_folder, os.path.dirname(data[0]['id']))
if not os.path.exists(root_path):
    os.makedirs(root_path)


def base64_to_image(base64_string, output_path):
    """Converts a base64 string to an image file.

    Args:
        base64_string: The base64 encoded string of the image.
        output_path: The path to save the image file.
    """
    try:
        image_data = base64.b64decode(base64_string)
        with open(output_path, 'wb') as image_file:
            image_file.write(image_data)
        #print(f"Image saved to {output_path}")
    except Exception as e:
        print(f"Error decoding base64 string: {e}")
for da in tqdm(data):
    json_data = {}
    json_data["id"] = da["id"]
    if da["image"] is not None:
        if '.jpg' not in da['id'] and '.png' not in da['id']:
            json_data["image"] = f"r1ov-image/{da['id']}.png"
        else:
            if da['id'][-3:] != 'jpg' and da['id'][-3:] != 'png':
                da['id'].replace('.jpg','')
                da['id'].replace('.png','')
                json_data["image"] = f"{da['id']}.png"
            else:
                json_data["image"] = f"{da['id']}"
        
        #da["image"].save(os.path.join(image_folder, json_data["image"]))
        base64_to_image(da["image"],os.path.join(image_folder, f"{da['id']}.png"))
        json_data["conversations"] = da["conversations"]
        converted_data.append(json_data)


# with open(json_file, "w") as f:
#     json.dump(converted_data, f, indent=4, ensure_ascii=False)

