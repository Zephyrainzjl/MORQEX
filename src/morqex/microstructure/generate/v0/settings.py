import yaml
import json
import os

class ConfigDict(dict):
    """
    一个辅助类，让字典可以通过点符号访问属性 (config.project.name)
    """
    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        val = self.get(attr)
        # 如果获取到的值还是字典，递归转换为 ConfigDict
        if isinstance(val, dict):
            return ConfigDict(val)
        return val

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        self.__delitem__(item)

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        super(ConfigDict, self).__setitem__(key, value)

class ConfigProcessor:
    def __init__(self, config_path):
        if isinstance(config_path, dict):
            self.config_path = None
            self._raw_data = config_path
            self.cfg = ConfigDict(config_path)
            return
        self.config_path = config_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")

        # 加载数据
        self._raw_data = self._load_yaml()
        # 转换为支持点操作的对象
        self.cfg = ConfigDict(self._raw_data)

    def _load_yaml(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def reload(self):
        """重新加载配置文件（用于在运行时修改yaml后刷新）"""
        print(f"Reloading config from {self.config_path}...")
        self._raw_data = self._load_yaml()
        self.cfg = ConfigDict(self._raw_data)

    def show_all(self):
        """格式化输出所有当前变量"""
        print("-" * 40)
        print(f"Current Configuration: {self.config_path}")
        print("-" * 40)
        # 使用 json dumps 进行漂亮的缩进打印
        print(json.dumps(self._raw_data, indent=4, ensure_ascii=False))
        print("-" * 40)

    def check_consistency(self):
        """
        根据 YAML 中的注释逻辑检查参数一致性
        """
        try:
            # 检查 1: d_feature_channels 第一维必须等于 img_channels
            img_ch = self.cfg.data.img_channels
            d_feat_first = self.cfg.network.d_feature_channels[0]
            if img_ch != d_feat_first:
                print(f"[Warning] Consistency Error: data.img_channels ({img_ch}) != network.d_feature_channels[0] ({d_feat_first})")
            else:
                print(f"[OK] Discriminator input channels match.")

            # 检查 2: g_feature_channels 第一维必须等于 z_channels
            z_ch = self.cfg.network.z_channels
            g_feat_first = self.cfg.network.g_feature_channels[0]
            if z_ch != g_feat_first:
                print(f"[Warning] Consistency Error: network.z_channels ({z_ch}) != network.g_feature_channels[0] ({g_feat_first})")
            else:
                print(f"[OK] Generator input (Z) channels match.")

            # 检查 3: g_feature_channels 最后一维必须等于 img_channels
            g_feat_last = self.cfg.network.g_feature_channels[-1]
            if img_ch != g_feat_last:
                print(f"[Warning] Consistency Error: data.img_channels ({img_ch}) != network.g_feature_channels[-1] ({g_feat_last})")
            else:
                print(f"[OK] Generator output channels match.")
        except Exception as e:
            print(f"[Error] Could not validate consistency: {e}")

    # 让 ConfigProcessor 实例本身也可以像字典一样被访问
    def __getattr__(self, name):
        return getattr(self.cfg, name)
