package.path = "%s?.lua;%s"%{ arg[0]:get_path(), package.path }
local utils = require "utils"
local brickpi = require "brickpi"
--
local images_dir = "%simages"%{ arg[0]:get_path() }
--
local filename = arg[1] or "nets/default.net"

-- BrickPi params

local trainer = util.deserialize(filename)
local thenet = trainer:get_component()

utils.setup_brickpi()
utils.do_action(utils.ACTION_STOP)

local finished = false
signal.register(signal.SIGINT, function() finished = true end)

utils.sleep(utils.SLEEP)
local clock = util.stopwatch()
while not finished do
  clock:reset()
  clock:go()
  --
  local img_path = utils.take_image(images_dir)
  local input    = utils.get_input_from_image_path(img_path)
  local output   = thenet:forward(input):get_matrix()
  local _,action = output:max()
  printf("%.4f %.4f %.4f :: %d\n",
         output:get(1,1), output:get(1,2), output:get(1,3),
         action)
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
