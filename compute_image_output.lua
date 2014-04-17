package.path = "%s?.lua;%s"%{ arg[0]:get_path(), package.path }
local utils = require "utils"
local image_filename = assert(arg[1], "Needs an image filename")
local filename = arg[2] or "nets/default.net"
local trainer = assert(util.deserialize(filename), "Unable to open " .. filename)
local thenet = trainer:get_component()
--
local clock = util.stopwatch()
clock:go()
local input = utils.get_input_from_image_path(image_filename)
local output = thenet:forward(input):get_matrix()
print(output)
print("TIME:",clock:read())
