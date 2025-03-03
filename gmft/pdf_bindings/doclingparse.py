
# PyMuPDF bindings
from typing import Generator
import PIL
import pypdfium2 as pdfium
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_parse.pdf_parsers import pdf_parser_v2
from PIL.Image import Image as PILImage
from PIL import ImageDraw
from pypdfium2 import PdfPage
from pathlib import Path
from gmft.common import Rect
from gmft.pdf_bindings.common import BasePage, BasePDFDocument, _infer_line_breaks
from docling.datamodel.base_models import Cell, Size, InputFormat
from docling.datamodel.document import InputDocument
from docling.backend.docling_parse_v2_backend  import DoclingParseV2DocumentBackend , DoclingParseV2PageBackend
from docling.utils.utils import create_file_hash, create_hash
from IPython.display import Image, display

class DoclingParserPDFPage(BasePage):
    
    def __init__(self, page: PdfPage,  filename: str, page_no: int, parser: pdf_parser_v2, document_hash: str ):
        self.page = page
        self.filename = filename


        parsed_page = parser.parse_pdf_from_key_on_page(document_hash, page_no)
        self.valid = "pages" in parsed_page and len(parsed_page["pages"]) == 1
        if self.valid:
            self._dpage = parsed_page["pages"][0]
        self.width = self.page.get_width()
        self.height = self.page.get_height()
        self._positions_and_text = [] # cache results, because this appears to be slow
        self._positions_and_text_and_breaks = []
        
        super().__init__(page_no)
    
    def get_positions_and_text(self) -> Generator[tuple[float, float, float, float, str], None, None]:
        cells_data = self._dpage["sanitized"]["cells"]["data"]
        cells_header = self._dpage["sanitized"]["cells"]["header"]

        #print(self._dpage)
        current_word = ""
        current_bbox = None
        parser_width = self._dpage["sanitized"]["dimension"]["width"]
        parser_height = self._dpage["sanitized"]["dimension"]["height"]
        result = []
        for i, cell_data in enumerate(cells_data):
            x0 = cell_data[cells_header.index("x0")]
            y0 = cell_data[cells_header.index("y0")]
            x1 = cell_data[cells_header.index("x1")]
            y1 = cell_data[cells_header.index("y1")]

            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0

            current_word = cell_data[cells_header.index("text")]
            # print(current_word)
            # print("\n")
            current_bbox = (x0*self.width / parser_width, self.height - y1 * self.height / parser_height, x1*self.width / parser_width, self.height - y0* self.height / parser_height)
            result.append((*current_bbox, current_word))
            yield *current_bbox, current_word
        
        self._positions_and_text = result


    
   
    def get_filename(self) -> str:
        return self.filename
    
    def get_image(self, dpi: int=None, rect: Rect=None) -> PILImage:
        if dpi is None:
            dpi = 72
        scale_factor = dpi / 72
        if rect is None:
            crop = (0, 0, self.width, self.height)
            xmin=0
            ymin=0
            xmax=self.width
            ymax=self.height
            bitmap = self.page.render(scale=scale_factor)
        else:
            # crop is "amount to cut off" from each side
            # left, bottom, right, top
            # crop = (rect.bbox[0], rect.bbox[1], self.page.get_width() - rect.bbox[0], self.page.get_height() - rect.bbox[1])
            xmin, ymin, xmax, ymax = rect.bbox
            # also remember that the origin is at the bottom left
            crop = (xmin, self.height - ymax, self.width - xmax, ymin)
            bitmap = self.page.render(scale=scale_factor, crop=crop)
        image = bitmap.to_pil()
        #ImageDraw.Draw(image).rectangle([xmin, ymin, xmax, ymax], outline="red")
        #display(image)
        return image
    
    def close(self):
        self.page.close()
        self.page = None
    
    # def __del__(self):
    #     if self.page is not None:
    #         self.close()
    
    def close_document(self):
        if self.page.parent:
            self.page.parent.close()
        self.page = None

    def _get_positions_and_text_and_breaks(self):
        """
        [Experimental] This is a generator that returns the positions and text of the page, as well as the breaks.
        """
        # cache, since it is slow
        if self._positions_and_text_and_breaks:
            for item in self._positions_and_text_and_breaks:
                yield item
            return
    
        # generate
        words = list(_infer_line_breaks(self.get_positions_and_text()))
        self._positions_and_text_and_breaks = words
        for item in words:
            yield item

class DoclingParseV2Document(BasePDFDocument):
    
    def __init__(self, filename: str):
        path = Path(filename)
        self.input_document = InputDocument(path,InputFormat.PDF,DoclingParseV2DocumentBackend)
        self.document_hash = self.input_document.document_hash
        self.doc = pdfium.PdfDocument(filename)
        self.parser = pdf_parser_v2("fatal")
        self.filename = filename
        success = self.parser.load_document(
                self.document_hash, str(path)
            )
       
    
    def get_page(self, n: int) -> BasePage:
        return DoclingParserPDFPage(self.doc[n],self.filename , n, self.parser, self.document_hash)
    
    def get_filename(self) -> str:
        return self.filename
    
    def __len__(self) -> int:
        return len(self.doc)
    
    def close(self):
        self.doc.close()