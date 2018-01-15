import os
import numpy as np

__all__ = ['FDDB', 'WIDER', 'CelebA', 'AFLW', 'L300WP']


class DataBase(object):
    def __init__(self, image_name_file):
        self.label_infos = self.get_labels(image_name_file)

    def get_labels(self, image_name_file):
        if not os.path.exists(image_name_file):
            raise ValueError("File is not exist.")
        with open(image_name_file, 'r') as f:
            lines = f.readlines()
        return lines

    def get_image_name(self, label_info):
        raise NotImplementedError()


class FDDB(DataBase):
    def __init__(self, dataset_path, image_name_file):
        super(FDDB, self).__init__(image_name_file)
        self.dataset_path = dataset_path

    # TODO: parser regular
    def label_parser(self, label_info):
        info = label_info.strip().split()
        image_name = info[0]
        if '.jpg' not in image_name:
            image_name += '.jpg'
        return os.path.join(self.dataset_path, image_name)

    def get_image_name(self, label_info, Full=1):
        image_name = label_info.strip().split()[0]
        if '.jpg' not in image_name:
            image_name += '.jpg'
        return os.path.join(self.dataset_path, image_name)

    def do_eval(self, image_name, results):
        #pure_image_name = os.path.splitext(image_name.split(self.dataset_path)[-1][1:])
        pure_image_name = image_name.strip()
        cal_boxes = results[0] if type(results) is tuple else results

        box_num = cal_boxes.shape[0]
        x = cal_boxes[:, 0]
        y = cal_boxes[:, 1]
        w = cal_boxes[:, 2] - cal_boxes[:, 0] + 1
        h = cal_boxes[:, 3] - cal_boxes[:, 1] + 1
        scores = cal_boxes[:, 4]

        write_result = '%s\n%d\n' % (pure_image_name, box_num)
        for i in range(box_num):
            # Face rect and score
            write_result += '%d %d %d %d %.4f\n' % (x[i], y[i], w[i], h[i], scores[i])
        return write_result


class WIDER(DataBase):
    def __init__(self, dataset_path, image_name_file, mode):
        super(WIDER, self).__init__(image_name_file)
        self.dataset_path = dataset_path
        self.mode = mode

    # TODO: parser regular
    def label_parser(self, label_info):
        label_info = label_info.strip().split(' ')
        image_name = os.path.join(self.dataset_path, label_info[0])
        boxes = np.array(map(float, label_info[1:]), dtype=np.float32).reshape(-1, 4)
        return {'image_name': image_name, 'gt_boxes': boxes}

    def get_image_name(self, label_info):
        image_name = label_info.strip().split()[0]
        return os.path.join(self.dataset_path, image_name)


class CelebA(DataBase):
    pass

class AFLW(DataBase):
    pass

class L300WP(DataBase):
    # http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
    # The synthesized large-pose face images from 300W.
    pass