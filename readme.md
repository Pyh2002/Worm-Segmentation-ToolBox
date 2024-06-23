# ML-based Worm Segmentation Toolbox

![sample_output](https://github.com/Pyh2002/Worm-Segmentation-ToolBox/assets/72658879/9876febd-ae4c-47bd-afee-9f3f7c387a06)

This toolbox is a worm tracker developed in [Eleni Gourgou's Lab](https://elenigourgou.engin.umich.edu/) at the University of Michigan, Ann Arbor. The project was primarily developed by Yuheng Pan and Adam Smith, under the supervision of Eleni Gourgou and her collaborator, Hongyi Xiao. It draws inspiration from the [tierpsy-tracker](https://github.com/Tierpsy/tierpsy-tracker) and Hongyi Xiao's [Bellybutton segmentation](https://pypi.org/project/bellybuttonseg/).

Our main goal is to track worms in more complex environments, particularly when the worm is surrounded by granular matter. The tool employs machine learning methods to track the worm, creating a binary mask over the worm, and uses Python libraries to derive specific parameters from the binary masked videos.

## Installation Instructions

### Binary mask generation

