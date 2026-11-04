"""Build Report 8 from saved results while retaining Report 7 formatting.

Requires the optional lxml document-authoring dependency. Only document.xml,
header1.xml and settings.xml are rewritten; the retained reference is unchanged.
"""
import argparse
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E
from progress_report8_content import write_report

ROOT = Path(__file__).resolve().parents[2]
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}


def tag(name):
    return f'{{{W}}}{name}'


def run(reference, output):
    with ZipFile(reference) as archive:
        original = {name: archive.read(name) for name in archive.namelist()}
    document = E.fromstring(original['word/document.xml'])
    body = document.find('w:body', NS)
    section = deepcopy(body.find('w:sectPr', NS))
    for child in list(body):
        body.remove(child)

    def paragraph(text, style='Normal', new_page=False):
        p = E.SubElement(body, tag('p'))
        props = E.SubElement(p, tag('pPr'))
        E.SubElement(props, tag('pStyle'), {tag('val'): style})
        if new_page:
            E.SubElement(props, tag('pageBreakBefore'))
        r = E.SubElement(p, tag('r'))
        E.SubElement(r, tag('t')).text = text

    def table(headers, rows, widths):
        if sum(widths) != 9936 or len(headers) != len(widths):
            raise ValueError('Table widths must fit retained page geometry.')
        tbl = E.SubElement(body, tag('tbl'))
        props = E.SubElement(tbl, tag('tblPr'))
        E.SubElement(props, tag('tblW'), {tag('w'): '9936', tag('type'): 'dxa'})
        E.SubElement(props, tag('tblLayout'), {tag('type'): 'fixed'})
        borders = E.SubElement(props, tag('tblBorders'))
        for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            E.SubElement(borders, tag(edge), {tag('val'): 'single', tag('sz'): '4', tag('color'): 'D9D9D9'})
        margins = E.SubElement(props, tag('tblCellMar'))
        for edge in ('top', 'left', 'bottom', 'right'):
            E.SubElement(margins, tag(edge), {tag('w'): '85', tag('type'): 'dxa'})
        grid = E.SubElement(tbl, tag('tblGrid'))
        for width in widths:
            E.SubElement(grid, tag('gridCol'), {tag('w'): str(width)})
        for index, values in enumerate([headers, *rows]):
            row = E.SubElement(tbl, tag('tr'))
            rowprops = E.SubElement(row, tag('trPr'))
            E.SubElement(rowprops, tag('cantSplit'))
            if index == 0:
                E.SubElement(rowprops, tag('tblHeader'))
            for column, value in enumerate(values):
                cell = E.SubElement(row, tag('tc'))
                cellprops = E.SubElement(cell, tag('tcPr'))
                E.SubElement(cellprops, tag('tcW'), {tag('w'): str(widths[column]), tag('type'): 'dxa'})
                E.SubElement(cellprops, tag('vAlign'), {tag('val'): 'center'})
                if index == 0:
                    E.SubElement(cellprops, tag('shd'), {tag('val'): 'clear', tag('fill'): 'E7EEF4'})
                p = E.SubElement(cell, tag('p'))
                pp = E.SubElement(p, tag('pPr'))
                E.SubElement(pp, tag('spacing'), {tag('after'): '0', tag('line'): '240', tag('lineRule'): 'auto'})
                E.SubElement(pp, tag('jc'), {tag('val'): 'left' if column == 0 else 'center'})
                r = E.SubElement(p, tag('r'))
                rp = E.SubElement(r, tag('rPr'))
                E.SubElement(rp, tag('sz'), {tag('val'): '20'})
                if index == 0:
                    E.SubElement(rp, tag('b'))
                E.SubElement(r, tag('t')).text = str(value)
        paragraph('')

    write_report(paragraph, table)
    body.append(section)
    header = E.fromstring(original['word/header1.xml'])
    for p in header.findall('w:p', NS):
        texts = p.findall('.//w:t', NS)
        text = ''.join(t.text or '' for t in texts)
        replacement = {'Progress Report 7': 'Progress Report 8', '10/28/2026': '11/04/2026'}.get(text)
        if replacement:
            texts[0].text = replacement
            for t in texts[1:]:
                t.text = ''
    settings = E.fromstring(original['word/settings.xml'])
    update = settings.find('w:updateFields', NS)
    if update is None:
        update = E.SubElement(settings, tag('updateFields'))
    update.set(tag('val'), 'true')
    changed = {name: E.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)
               for name, tree in [('word/document.xml', document), ('word/header1.xml', header), ('word/settings.xml', settings)]}
    with ZipFile(output, 'w') as archive:
        for name, data in original.items():
            archive.writestr(name, changed.get(name, data))
    with ZipFile(output) as archive:
        assert all(archive.read(name) == data for name, data in original.items() if name not in changed)
    print(f'Saved {output}. Retained template and preserve-only package parts are unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT / 'docs/Progress_Reports/Progress_Report_7.docx')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/Progress_Reports/Progress_Report_8.docx')
    args = parser.parse_args()
    run(args.reference, args.output)
