from anytree import Node, RenderTree
from anytree.exporter import DotExporter
root = Node("root") #root
a = Node("class A", parent=root)
b = Node("class B", parent=root)
c = Node("class C", parent=root)
parentList = [a,b,c]
for i in range(3):
  for j in range(5):
    node = Node(i*5+j, parent=parentList[i])
DotExporter(root).to_picture("ceo.png")