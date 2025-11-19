[robot]
# ángulos en grados, como en --degrees
angle_unit = "deg"

[[robot.joint]]
type = "R"
L = 5.0
theta = 0.0
extension = 0.0
limits = [0.0, 180.0]

[[robot.joint]]
type = "R"
L = 5.0
theta = 0.0
extension = 0.0
limits = [0.0, 180.0]

[[robot.joint]]
type = "R"
L = 5.0
theta = 0.0
extension = 0.0
limits = [0.0, 180.0]

[[robot.joint]]
type = "P"
L = 0.0
theta = 0.0      # orientación del eje prismático
extension = 0.0  # extensión inicial
limits = [0.0, 10.0]