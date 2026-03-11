import time
from agent_system.environments.env_package.habitat_sim.utils.hm3d_unoccluded_utils import HM3DMeshExtractor, HM3DSemanticParser
from agent_system.environments.env_package.habitat_sim.utils.habitat_utils import create_hm3d_simulator

scene_id = "/data/tct/habitat/data/hm3d/train/00009-vLpv2VX547B/vLpv2VX547B.basis.glb"

sim = create_hm3d_simulator("HM3D", scene_id)

print("#######################################")
wait_time = 10
time.sleep(wait_time)

