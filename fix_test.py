import re

with open("Src/TestTurboQuant/Program.cs", "r") as f:
    content = f.read()

# Replace `TurboQuant.Create` initialization
new_init = "var tq = new TurboQuant(dim, 42);"
content = re.sub(r'var tq = TurboQuant.Create\(.*?\);', new_init, content)

with open("Src/TestTurboQuant/Program.cs", "w") as f:
    f.write(content)
