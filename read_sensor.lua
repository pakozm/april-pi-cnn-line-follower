package.path = "%s?.lua;%s"%{ arg[0]:get_path(), package.path }
local utils = require "utils"
local brickpi = require "brickpi"

-- QLearning parameters
local PENALTY          =  -1
local REWARD           =   1

-- MAIN
local sensor = utils.sensor(utils.LIGHT_SENSOR,
                            REWARD, PENALTY)

utils.setup_brickpi()
utils.do_action(utils.ACTION_STOP)
sensor:calibrate()

utils.sleep(utils.SLEEP)
local clock = util.stopwatch()
while true do
  clock:reset()
  clock:go()
  --
  local reward,value = sensor:compute_reward()
  print(reward, value, sensor.BLACK_LOW, sensor.BLACK_HIGH)
  utils.do_until(brickpi.update)
  --
  clock:stop()
  local t1,t2 = clock:read()
  local extra_sleep = utils.SLEEP - t1
  if extra_sleep > 0 then
    utils.sleep(extra_sleep)
  end
end
