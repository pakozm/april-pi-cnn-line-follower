local offline_controller
local brickpi = {
  sensorValue = function()
    return offline_controller:get_info().sensor
  end
}
setmetatable(brickpi,
             {
               __index = function() return function() return true end end
             })
package.preload.brickpi = function() return brickpi end
package.path = "%s?.lua;%s"%{ arg[0]:get_path(), package.path }
local utils = require "utils"
--
offline_controller = utils.offline_controller("%sdata/"%{arg[0]:get_path()})
--
local filename            = arg[1] or "nets/default.net"
local out_filename        = arg[2] or "nets/last.net"

-- QLearning parameters
local DISCOUNT         = 0.6
local PENALTY          =  -0.5
local REWARD           =   2.0

-- MAIN
local trainer = utils.trainer(filename, DISCOUNT)
local sensor = utils.sensor(utils.LIGHT_SENSOR,
                            REWARD, PENALTY)
local finished = false
signal.register(signal.SIGINT, function() finished = true end)

while offline_controller:next() and not finished do
  collectgarbage("collect")
  local img_path = offline_controller:get_input_path()
  local info = offline_controller:get_info()
  sensor.BLACK_MEAN = info.mean
  sensor.BLACK_V = info.var
  sensor.slope = (REWARD - PENALTY) / (sensor.BLACK_V * 2)
  trainer.prev_action = info.action
  local action,qs = trainer:one_step(img_path, sensor)
  trainer:save(out_filename)
  local w1 = trainer.tr:weights("w1")
  local sz = math.sqrt(w1:dim(2))
  local img = ann.connections.input_filters_image(w1, {sz,sz})
  ImageIO.write(img,"data/wop.png")
end
