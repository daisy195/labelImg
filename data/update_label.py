import xml.etree.ElementTree as ET
import glob


def update(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls = obj.find('name')
        if cls.text in ["top_left", "top_right", "bottom_right", "bottom_left"]:
            cls.text = "corner"

    tree.write(xml_file)


xmls = [f for f in glob.glob("./labels/cmnd/*.xml", recursive=True)]

for xml in xmls:
    update(xml_file=xml)
