import io
import pandas as pd
# from tabled.extract import extract_tables
from tabled.fileinput import load_pdfs_images
from tabled.inference.models import load_detection_models, load_recognition_models
from tabled.formats import formatter


from typing import List

from tabled.assignment import assign_rows_columns
from tabled.inference.detection import detect_tables
from tabled.inference.recognition import get_cells, recognize_tables
from tabled.schema import ExtractPageResult
from surya.settings import settings as surya_settings

from gmft.detectors.common import CroppedTable
from gmft.formatters.common import BaseFormatter, FormattedTable





def extract_tables(images, highres_images, text_lines, det_models, rec_models, skip_detection=False, detect_boxes=False) -> List[ExtractPageResult]:
    if not skip_detection:
        table_imgs, table_bboxes, table_counts = detect_tables(images, highres_images, det_models)
    else:
        table_imgs = highres_images
        table_bboxes = [[0, 0, img.size[0], img.size[1]] for img in highres_images]
        table_counts = [1] * len(highres_images)

        # table_imgs = 1148x254 (it has been cropped)


    table_text_lines = []
    highres_image_sizes = []
    for i, tc in enumerate(table_counts):
        table_text_lines.extend([text_lines[i]] * tc)
        highres_image_sizes.extend([highres_images[i].size] * tc)

    cells, needs_ocr = get_cells(table_imgs, table_bboxes, highres_image_sizes, table_text_lines, det_models[:2], detect_boxes=detect_boxes)

    table_rec = recognize_tables(table_imgs, cells, needs_ocr, rec_models)
    cells = [assign_rows_columns(tr, im_size) for tr, im_size in zip(table_rec, highres_image_sizes)]

    results = []
    counter = 0
    for count in table_counts:
        page_start = counter
        page_end = counter + count
        results.append(ExtractPageResult(
            table_imgs=table_imgs[page_start:page_end],
            cells=cells[page_start:page_end],
            rows_cols=table_rec[page_start:page_end]
        ))
        counter += count

    assert len(results) == len(images)
    return results

def _extract_tables_patched(images, highres_images, table_bboxes, text_lines, 
                            det_models, rec_models, detect_boxes=False) -> List[ExtractPageResult]:
    # if not skip_detection:
        # table_imgs, table_bboxes, table_counts = detect_tables(images, highres_images, det_models)
    # else:
    # crop image according to table_bboxes
    # table_imgs = highres_images
    table_imgs = [img.crop(bbox) for img, bbox in zip(highres_images, table_bboxes)]

    # table_bboxes = [[0, 0, img.size[0], img.size[1]] for img in highres_images]
    table_counts = [1] * len(highres_images)

    table_text_lines = []
    highres_image_sizes = []
    for i, tc in enumerate(table_counts):
        table_text_lines.extend([text_lines[i]] * tc)
        highres_image_sizes.extend([highres_images[i].size] * tc)

    cells, needs_ocr = get_cells(table_imgs, table_bboxes, highres_image_sizes, table_text_lines, det_models[:2], detect_boxes=detect_boxes)

    table_rec = recognize_tables(table_imgs, cells, needs_ocr, rec_models)
    cells = [assign_rows_columns(tr, im_size) for tr, im_size in zip(table_rec, highres_image_sizes)]

    results = []
    counter = 0
    for count in table_counts:
        page_start = counter
        page_end = counter + count
        results.append(ExtractPageResult(
            table_imgs=table_imgs[page_start:page_end],
            cells=cells[page_start:page_end],
            rows_cols=table_rec[page_start:page_end]
        ))
        counter += count

    assert len(results) == len(images)
    return results


# det models: EfficientViTForSemanticSegmentation, SegformerImageProcessor, EfficientViTForSemanticSegmentation, SegformerImageProcessor
# rec models: TableRecEncoderDecoderModel, SuryaProcessor, OCREncoderDecoderModel, SuryaProcessor

# images: list[PIL.Image.Image] 816x1056
# highres_images: list[PIL.Image.Image] 1632x2112
# names: list[str]
# text_lines: list[dict] one for each image (page)
# [{'blocks': [...], 'page': 0, 'rotation': 0, 'bbox': [...], 'width': 612, 'height': 792}]
# text_lines[#].blocks[#] = {'bbox': (156.4, 72.9, 438.6, 81.8), 'lines': LinesDict}
# LinesDict = [{'bbox': list[4], 'spans': [
# {'chars': [{'char': 'T', 'bbox': BBox}], 
# 'font': {'size': float, 'weight': float, 'name': str, 'flags': int},
# 'rotation': float,
# 'bbox': Bbox,
# 'text': str,
# 'char_start_idx': int,
# 'char_end_idx': int
# }
# ]}]

# note that the priority is as follows:
# 1. if the 
import json

class SuryaTabledFormatter(BaseFormatter):
    def __init__(self):
        self.det_models, self.rec_models = load_detection_models(), load_recognition_models()

# def main():
    
    # print(gmft_dict['bbox'])
    def extract(self, ct: CroppedTable):

        gmft_dict = ct.bbox

        import os
        if os.path.isfile(ct.page.get_filename()):
            print("File exists: ", ct.page.get_filename())
        else:
            print("File does not exist: ", ct.page.get_filename())
            # we will have to mock up the format of text_lines
            raise NotImplementedError

        # images, highres_images, names, text_lines = load_pdfs_images('support_arena/spantest.pdf') # (instant)
        # images, highres_images, names, text_lines = load_pdfs_images('support_arena/spantest.pdf') # (instant)
        images, highres_images, names, text_lines = load_pdfs_images(ct.page.get_filename(), max_pages=1, start_page=ct.page.page_number) # (instant)

        pnums = []
        prev_name = None
        for i, name in enumerate(names):
            if prev_name is None or prev_name != name:
                pnums.append(0)
            else:
                pnums.append(pnums[-1] + 1)

            prev_name = name
            
        page_results = extract_tables(images, highres_images, text_lines, self.det_models, self.rec_models)

        # need to multiply by IMAGE_DPI_HIGHRES / 72 = 192 / 72 

        # target: [[222, 228, 1370, 482]]
        # actual: [[238, 254, 1336, 464]]
        fixed_box = [int(x * surya_settings.IMAGE_DPI_HIGHRES / 72) for x in gmft_dict] # ['bbox']]
        fixed_boxes = [[x0 - 30, y0 - 30, x1 + 30, y1 + 30] for x0, y0, x1, y1 in [fixed_box]] # add some padding, just like in tatr
        # fixed_boxes = [[222, 228, 1370, 482]]
        page_results = _extract_tables_patched(images, highres_images, fixed_boxes, text_lines, self.det_models, self.rec_models)

        # print(page_results)

        for name, pnum, result in zip(names, pnums, page_results):
            for i in range(result.total):
                page_cells = result.cells[i]
                # page_rc = result.rows_cols[i]
                # img = result.table_imgs[i]

                # base_path = os.path.join(out_folder, name)
                # os.makedirs(base_path, exist_ok=True)

                formatted_result, ext = formatter('csv', page_cells)
                df = pd.read_csv(io.StringIO(formatted_result))
                
                # base_name = f"page{pnum}_table{i}"
                # print(df)
        
                return FormattedTable(ct, df)
                # TODO: multiple tables
        
        return None
                

# https://github.com/VikParuchuri/tabled/blob/master/extract.py
# main()


if __name__ == '__main__':
    stf = SuryaTabledFormatter()
    # gmft_dict = json.load(open('test/outputs/pubt/pubt_p4.info'))
    from gmft_pymupdf import PyMuPDFDocument
    doc = PyMuPDFDocument('support_arena/spantest.pdf')
    with open('test/outputs/pubt/pubt_p4.info') as f:
        ct = CroppedTable.from_dict(json.load(f), doc[0])
        ct.page.page_number = 0
        ft = stf.extract(ct)
        print(ft.df())