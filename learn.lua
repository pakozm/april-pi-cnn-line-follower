package.path = "%s?.lua;%s"%{ arg[0]:get_path(), package.path }
local utils = require "utils"
local brickpi = require "brickpi"
--
local images_dir = "%simages"%{ arg[0]:get_path() }
--
local filename            = arg[1] or "nets/default.net"
local out_filename        = arg[2] or "nets/last.net"

-- QLearning parameters
local HISTORY_DISCOUNT = 0.5
local DISCOUNT         = 0.9
local ALPHA            = 0.1
local PENALTY          =  -1
local REWARD           =   1

--
local save = false -- save snapshot

-- FUNCTIONS

local idx=1
local function save_snapshot(trainer,sensor)
  if save then
    ImageIO.write(trainer.input_img, "data/input%06d.png"%{ idx })
    local f = io.open("data/sensor%06d.txt"%{ idx - 0 }, "w")
    fprintf(f, "%d ( %d %d )\n", sensor.value, sensor.BLACK_LOW, sensor.BLACK_HIGH)
    f:close()
    idx=idx+1
  end
end

-- MAIN
os.execute("rm -f data/*")
local trainer = utils.trainer(filename, ALPHA, DISCOUNT)
local sensor = utils.sensor(utils.LIGHT_SENSOR,
                            REWARD, PENALTY, HISTORY_DISCOUNT)

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
  --
  local img_path = utils.take_image(images_dir)
  local action = trainer:one_step(img_path, sensor)
  save_snapshot(trainer, sensor)
  trainer:save(out_filename)
  utils.do_action(action)
  utils.do_until(brickpi.update)
  --
  clock:stop()
  local t1,t2 = clock:read()
  local extra_sleep = utils.SLEEP - t1
  if extra_sleep > 0 then
    utils.sleep(extra_sleep)
  end
end
