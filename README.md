# Plastic-detection
This is one of the high school artificial intelligence projects, where yolov5 and my custom dataset were used to train a model.

## otivation
 Korea boasts a recycling rate well above the **OECD average recycling rate**. However, if you look at the **real recycling rate** on the other side, it is similar to the **OECD average recycling rate**. Experts say that the reason for this is the difficulty of classification due to the various types of plastics. Therefore, it is more economically beneficial to discard plastics than to recycle them. After understanding these problems, I thought, 'What if artificial intelligence classifies plastics?'

## Perform
 First, a total of four classes were set for PE, PET, PP, and PS. After that, pictures were taken and data were collected. Insufficient data were supplemented in [AIHub](https://www.aihub.or.kr/) to compose the data. And I finally composed my custom dataset by working on data labeling in [Roboflow](https://roboflow.com/). Since the basic data is 2479 sheets, which is very small to proceed with model learning, it has been augmented to a total of 7377 sheets through the **Data Augmentation** process.

## License

© 2022, Plastic-detection. Released under [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).

**Plastic-detection** repository is authored and maintained by [@CharlesbrownK](https://github.com/CharlesbrownK).
