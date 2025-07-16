import os
import sys
import xml
import xml.etree.ElementTree

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)


junit_dir = os.path.join(__PROJECT_ROOT__, "docs/source/test-report/all_reports")
junit_file = os.path.join(junit_dir, "junit-ubuntu-latest-3.10.xml")

tree = xml.etree.ElementTree.parse(junit_file)
root = tree.getroot()


print(type(tree))
# print(type(root))

print(tree.findall("./testsuite/*"))



# print()
# # print()
# # print()
# print(dir(root))
# # print(root.keys())
# # print(root.get("name"))
# # 
# # for it in tree.iter():
# #     print(type(it))
# 
# print(root.items())
# print(root.attrib)
# # print(root.keys())

# print(root.findall("./testsuitess"))



