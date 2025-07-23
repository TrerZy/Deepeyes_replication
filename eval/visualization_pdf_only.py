import os
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from matplotlib.backends.backend_pdf import PdfPages
import re
from tqdm import tqdm

def format_print_str(long_str, line_length=175):
    return '\n'.join([long_str[i:i+line_length] for i in range(0, len(long_str), line_length)])

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 > x1 and y2 > y1:
        inter_area = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter_area / (box1_area + box2_area - inter_area)
    return 0.

def draw_boxes_on_image(image, boxes, color=(255, 0, 0), width=3):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        draw.rectangle(box, outline=color, width=width)
        draw.text((box[0], box[1]), str(i + 1), fill=color)
    return image

def extract_zoom_boxes(conversation):
    zoom_boxes = []
    seen = set()
    for conv in conversation:
        if conv['role'] == 'assistant':
            content = conv['content']
            if isinstance(content, str):
                content = content
            elif isinstance(content, list):
                for _content in content:
                    if _content['type'] == 'text':
                        content = _content['text']
            else:
                continue

            pattern = r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]'
            matches = re.findall(pattern, content)

            for match in matches:
                box = tuple(int(num) for num in match)
                if box not in seen:
                    seen.add(box)
                    zoom_boxes.append(list(box))

    return zoom_boxes

def save_crops(image, boxes, save_dir, gt=False, save=True):
    crops = []
    for i, box in enumerate(boxes):
        crop = image.crop(box)
        crops.append(crop)
        if save:
            if gt:
                crop.save(os.path.join(save_dir, f"gt_crop_{i+1}.jpg"))
            else:
                crop.save(os.path.join(save_dir, f"zoom_crop_{i+1}.jpg"))
    return crops

def save_text_summary(save_dir, image_name, question, answer, pred_ans, acc, conversation):
    with open(os.path.join(save_dir, "summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Image: {image_name}\n")
        f.write(f"Question: {question}\n")
        f.write(f"GT Answer: {answer}\n")
        f.write(f"Predicted Answer: {pred_ans}\n")
        f.write(f"{'Success!' if acc else 'Fail!'}\n\n")
        f.write("Conversation:\n")
        for conv in conversation:
            role = conv['role'].upper()
            if isinstance(conv['content'], list):
                for item in conv['content']:
                    if item.get('type') == 'text':
                        f.write(f"{role}: {format_print_str(item['text'])}\n\n")
            elif isinstance(conv['content'], str):
                f.write(f"{role}: {format_print_str(conv['content'])}\n\n")

def save_pdf(pdf_name, image, image_with_gt, image_with_zoom, gt_crops, zoom_crops, image_name, question, answer, pred_ans, acc, conversation):

    with PdfPages(pdf_name) as pdf:

        # Summary page
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis('off')
        summary = f"Image: {image_name}\nQuestion: {question}\nGT Answer: {answer}\nPredicted: {pred_ans}\n{'Success!' if acc else 'Fail!'}\n\n"
        ax.text(0.01, 0.99, summary, ha='left', va='top', wrap=True, fontsize=9)

        pdf.savefig(fig)
        plt.close(fig)

        # Add images to PDF
        def add_image(img, title):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

        add_image(image, "Original Image")
        add_image(image_with_gt, "Original with GT Boxes")
        for i, crop in enumerate(gt_crops):
            add_image(crop, f"gt Crop {i+1}")
        add_image(image_with_zoom, "Original with Zoom-In Boxes")
        for i, crop in enumerate(zoom_crops):
            add_image(crop, f"Zoom-In Crop {i+1}")

        # Text page
        conv_text = "\n"
        for conv in conversation:
            role = conv['role'].upper()
            if isinstance(conv['content'], list):
                for item in conv['content']:
                    if item.get('type') == 'text':
                        conv_text += f"{role}: {item['text']}\n\n"
            elif isinstance(conv['content'], str):
                conv_text += f"{role}: {conv['content']}\n\n"
        conv_text = format_print_str(conv_text, line_length=80)
        lines = conv_text.split('\n')

        lines_per_page = 58
        total_pages = (len(lines) + lines_per_page - 1) // lines_per_page
        
        for page_num in range(total_pages):

            fig, ax = plt.subplots(figsize=(8, 12.5))
            ax.axis('off')

            start_idx = page_num * lines_per_page
            end_idx = min(start_idx + lines_per_page, len(lines))
            page_lines = lines[start_idx:end_idx]

            if total_pages > 1:
                ax.text(0.5, 0.01, f"Page {page_num+1}/{total_pages}", 
                        ha='center', va='bottom', fontsize=8)

            page_text = '\n'.join(page_lines)
            if page_num == 0:
                head = "Conversation:\n\n"
                page_text = head + page_text
            ax.text(0.01, 0.99, page_text, ha='left', va='top', wrap=True, fontsize=9)
            
            # 保存当前页
            pdf.savefig(fig)
            plt.close(fig)

def process_sample(image_name, data, root_path, save_root, pdf_only=False):
    image_path = os.path.join(root_path, image_name)
    json_path = os.path.join(root_path, image_name.replace('.jpg', '.json'))
    
    if not os.path.exists(image_path) or not os.path.exists(json_path):
        print(f"[WARN] Missing image or json for: {image_name}")
        return

    question = data['question']
    answer = data['answer']
    pred_ans = data['pred_ans']
    acc = data['acc']
    conversation = data['pred_output']

    status = f"success" if acc else "fail"
    save_root = os.path.join(save_root, status)
    os.makedirs(save_root, exist_ok=True)

    if pdf_only:
        save_name = image_name.replace(".jpg", ".pdf")
        pdf_name = os.path.join(save_root, save_name)
        save_dir = ""
    else:
        save_name = image_name.replace(".jpg", "")
        save_dir = os.path.join(save_root, save_name)
        pdf_name = os.path.join(save_dir, "summary.pdf")
        os.makedirs(save_dir, exist_ok=True)

    # Load image and GT boxes
    image = Image.open(image_path)
    gt_data = json.load(open(json_path, 'r'))
    gt_boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in gt_data['bbox']]

    # Draw GT boxes
    image_with_gt = draw_boxes_on_image(image, gt_boxes, color=(0, 255, 0))
    # image_with_gt.save(os.path.join(save_dir, "original_with_gt_boxes.jpg"))

    # Extract and draw zoom-in boxes
    zoom_boxes = extract_zoom_boxes(conversation)
    image_with_zoom = draw_boxes_on_image(image, zoom_boxes, color=(255, 0, 0))
    # image_with_zoom.save(os.path.join(save_dir, "original_with_zoom_boxes.jpg"))

    # Save gt crops
    gt_crops = save_crops(image, gt_boxes, save_dir, gt=True, save=False)

    # Save zoom-in crops
    zoom_crops = save_crops(image, zoom_boxes, save_dir, gt=False, save=False)

    # # Save raw image
    # image.save(os.path.join(save_dir, "original.jpg"))

    # # Save text summary
    # save_text_summary(save_dir, image_name, question, answer, pred_ans, acc, conversation)
    
    # save PDF
    save_pdf(pdf_name, image, image_with_gt, image_with_zoom, gt_crops, zoom_crops, image_name, question, answer, pred_ans, acc, conversation)

def main():
    root_path = '/cluster/home3/zhaoyutian/datasets/VstarBench'
    json_path = '/cluster/home3/zhaoyutian/code/deepeyes/DeepEyes/results/Vstar_result/deepeyes/result_relative_position_deepeyes_acc.jsonl'
    save_root = '/cluster/home3/zhaoyutian/code/deepeyes/DeepEyes/results/Vstar_result/deepeyes/visualization_pdf_only'

    if 'direct_attributes' in json_path:
        root_path = os.path.join(root_path, 'direct_attributes')
        save_root = os.path.join(save_root, 'direct_attributes')
    else:
        root_path = os.path.join(root_path, 'relative_position')
        save_root = os.path.join(save_root, 'relative_position')
    os.makedirs(save_root, exist_ok=True)

    with open(json_path, 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]

    line_map = {line['image']: line for line in lines}
    image_list = list(line_map.keys())

    progress_bar = tqdm(
        total=len(image_list),
        desc="Processing Images",
        unit="image",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    success_count = 0
    failed_count = 0

    for image_name in image_list:
        try:
            data = line_map[image_name]
            process_sample(image_name, data, root_path, save_root, pdf_only=True)
            success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {image_name}: {e}")
            failed_count += 1
        finally:
            progress_bar.update(1)

if __name__ == "__main__":
    main()
