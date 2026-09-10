import matplotlib.pyplot as plt
import numpy as np


class BrainActivityRenderer:
    def __init__(self, points, camera_points, network_indices):
        self.points = points
        self.network_indices = network_indices

        self.center = np.median(camera_points, axis=0)

        low = np.percentile(camera_points, 5, axis=0)
        high = np.percentile(camera_points, 95, axis=0)

        self.span = high - low
        self.half = self.span.max() * 0.55 / 2

    def build_colors(
        self,
        voltage,
        direct_input,
        ever_spiked,
        recent_spikes,
    ):
        indices = self.network_indices

        voltage = voltage[indices]
        direct = direct_input[indices]
        spiked = ever_spiked[indices]
        recent = recent_spikes[indices]

        activity = np.clip(
            (voltage + 65.0) / 15.0,
            0.0,
            1.0,
        )

        colors = np.full(
            (len(indices), 3),
            0.25,
            dtype=float,
        )

        # Прямой stimulus
        colors[direct] = [0.15, 0.55, 1.0]

        # Возбудился через сеть
        indirect = (~direct) & (
            spiked | (activity > 0.1)
        )
        colors[indirect] = [1.0, 0.45, 0.05]

        # Недавний spike
        colors[recent] = [1.0, 1.0, 1.0]

        return colors

    def render(self, colors, output_path=None):
        fig = plt.figure(
            figsize=(10, 8),
            facecolor="black",
        )

        ax = fig.add_subplot(
            111,
            projection="3d",
            facecolor="black",
        )

        ax.scatter(
            self.points[:, 0],
            self.points[:, 1],
            self.points[:, 2],
            s=3,
            c=colors,
            alpha=0.9,
            #depthshade=False, убирает глубину
        )

        ax.set_proj_type("ortho")
        ax.set_box_aspect(self.span)
        ax.view_init(elev=90, azim=-90)

        ax.set_xlim(
            self.center[0] - self.half,
            self.center[0] + self.half,
        )
        ax.set_ylim(
            self.center[1] - self.half,
            self.center[1] + self.half,
        )
        ax.set_zlim(
            self.center[2] - self.half,
            self.center[2] + self.half,
        )

        ax.set_axis_off()

        plt.tight_layout()

        # Старый рабочий вариант для просмотра одного кадра.
        # plt.show()

        if output_path is None:
            plt.show()
        else:
            fig.savefig(
                output_path,
                facecolor="black",
            )
            plt.close(fig)