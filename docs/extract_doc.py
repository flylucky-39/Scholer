"""Simple .doc text extractor - scans for UTF-16LE text."""
import olefile
import struct
import re

def extract_text_simple(filepath):
    ole = olefile.OleFileIO(filepath)
    doc_data = ole.openstream('WordDocument').read()
    table_data = ole.openstream('1Table').read()
    ole.close()

    all_text = []

    # Scan all data for UTF-16LE encoded text
    for data in [doc_data, table_data]:
        text = data.decode('utf-16-le', errors='ignore')
        # Keep only printable text
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # Replace runs of whitespace
        text = re.sub(r'\s+', ' ', text)
        # Only keep chunks with Chinese or meaningful content
        if len(text) > 20:
            all_text.append(text)

    result = '\n'.join(all_text)
    # Extract only the useful parts
    # Look for actual Chinese/English content lines
    lines = result.split('\n')
    good_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 10 and not line.startswith('Root') and not line.startswith('\x00'):
            good_lines.append(line)

    return '\n'.join(good_lines[:200])  # First 200 lines

if __name__ == '__main__':
    text = extract_text_simple('E:/Study/FSOD_VLM/fsod/docs/221349391900.doc')
    with open('E:/Study/FSOD_VLM/fsod/docs/doc_output.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print('Done - check doc_output.txt')
