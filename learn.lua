package.path = "%s?.lua;%s"%{ arg[0]:get_path(), package.path }
local utils = require "utils"
local brickpi = require "brickpi"
--
local images_dir = "%simages"%{ arg[0]:get_path() }
--
local filename            = arg[1] or "nets/default.net"
local out_filename        = arg[2] or "nets/last.net"

-- QLearning parameters
local DISCOUNT         = 0.6
local PENALTY          =  -0.5
local REWARD           =   2.0

--
local save = true -- save snapshot

-- FUNCTIONS

local idx=1
local function save_snapshot(trainer,sensor,action)
  if save and idx < 100000 then
    ImageIO.write(trainer.input_img, "data/input%06d.png"%{ idx })
    local f = io.open("data/info%06d.txt"%{ idx - 1 }, "w")
    fprintf(f, "%d %d ( %d +- %d )\n",
            action, sensor.value, sensor.BLACK_MEAN, sensor.BLACK_V)
    f:close()
    idx=idx+1
  end
end

-- MAIN
os.execute("rm -f data/*")
local trainer = utils.trainer(filename, DISCOUNT)
local sensor = utils.sensor(utils.LIGHT_SENSOR,
                            REWARD, PENALTY)

utils.setup_brickpi()
utils.do_action(utils.ACTION_STOP)
sensor:calibrate()

local finished = false
signal.register(signal.SIGINT, function() finished = true end)

utils.sleep(utils.SLEEP)
local clock = util.stopwatch()
while not finished do
  clock:reset()
  clock:go()
  collectgarbage("collect")
  --
  local img_path = utils.take_image(images_dir)
  local action = trainer:one_step(img_path, sensor)
  save_snapshot(trainer, sensor, action)
  trainer:save(out_filename)
  utils.do_action(action)
  utils.do_until(brickpi.update)
  local img = ann.connections.input_filters_image(trainer.tr:weights("w1"),
                                                  {24, 24})
  ImageIO.write(img,"data/wop.png")
  --
  clock:stop()
  local t1,t2 = clock:read()
  local extra_sleep = utils.SLEEP - t1
  if extra_sleep > 0 then
    utils.sleep(extra_sleep)
  end
end

brickpi.sensorType(utils.LIGHT_SENSOR,brickpi.TYPE_SENSOR_LIGHT_OFF)
brickpi.setupSensors()
