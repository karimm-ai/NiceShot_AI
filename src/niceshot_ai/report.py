import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class ReportMaker:
    def __init__(self, output_dir, csv_file, events_config, report_config, bucket_len):
        self.output_dir = output_dir
        self.csv_file = csv_file
        self.events_config = events_config
        widths = [c["width"] for c in report_config["charts"]]
        self.bucket_len = bucket_len

        # Create ONE figure with 3 axes
        self.fig, self.axes = plt.subplots(
            1, len(report_config["charts"]),
            figsize=(sum(widths) * 3, 6),   # dynamic width
            gridspec_kw={"width_ratios": widths}
        )

        if len(report_config["charts"]) == 1:
            self.axes = [self.axes]

        # Set global background
        self.fig.patch.set_facecolor(report_config["color_pallete"]["chart_color"])

        self.current_axis = 0


    def total_events_count_chart(self, color_pallete: dict, width, height):
        df = pd.read_csv(self.csv_file)
        event_counts = df["Event"].value_counts()
        events = []
        for event, _ in self.events_config.items():
            events.append(event)

        counts = [event_counts.get(e, 0) for e in events]

        self.axes[self.current_axis].set_facecolor(color_pallete["chart_color"])
        p = self.axes[self.current_axis].bar(events, counts, color=color_pallete["elements_color"])
        self.axes[self.current_axis].bar_label(p, label_type='edge', color=color_pallete["font_color"])
        self.axes[self.current_axis].set_title("Session Totals", color=color_pallete["font_color"], loc='left')
        self.axes[self.current_axis].tick_params(axis='x', colors=color_pallete["font_color"])
        self.axes[self.current_axis].tick_params(axis='y', colors=color_pallete["font_color"])

        self.current_axis+=1

    
    def kd_ratio_card(self, color_pallete: dict, width, height):
        if "Death" in self.events_config.keys() and "Kill" in self.events_config.keys():
            df = pd.read_csv(self.csv_file)
            kills = (df["Event"] == "Kill").sum()
            deaths = (df["Event"] == "Death").sum()

            kd_ratio = kills / deaths if deaths != 0 else kills

            self.axes[self.current_axis].axis("off")
            self.axes[self.current_axis].text(0.5, 0.65, "KD Ratio",
                    fontsize=18,
                    ha="center",
                    weight="bold",
                    color=color_pallete["font_color"])
            self.axes[self.current_axis].text(0.5, 0.40, f"{kd_ratio:.2f}",
                    fontsize=28,
                    ha="center",
                    color=color_pallete["font_color"])
            self.axes[self.current_axis].text(0.5, 0.15, f"{kills} Kills / {deaths} Deaths",
                    fontsize=12,
                    ha="center",
                    color=color_pallete["font_color"])
            self.axes[self.current_axis].tick_params(axis='x', colors=color_pallete["font_color"])
            self.axes[self.current_axis].tick_params(axis='y', colors=color_pallete["font_color"])

            self.current_axis+=1
        
        else:
            print("Metrics are not available!")


    def kd_timeline_chart(self, color_pallete: dict, width, height):
        df = pd.read_csv(self.csv_file)

        bucket_seconds = self.bucket_len*60
        print(bucket_seconds)
        df["Seconds"] = pd.to_timedelta(df["Timestamp"]).dt.total_seconds()
        df = df.sort_values("Seconds")
        print(df)
        df["Bucket"] = (df["Seconds"] // bucket_seconds).astype(int)
        df.to_csv(f"{self.output_dir}/timestamp_sorted.csv", index=False)

        # Cumulative kills per bucket
        kills_per_bucket = (
            df[df["Event"].str.lower() == "kill"]
            .groupby("Bucket")["Event"]
            .count()
            .cumsum()
            .reset_index(name="Cumulative_Kills")
        )
        print(kills_per_bucket)

        # Cumulative deaths per bucket
        deaths_per_bucket = (
            df[df["Event"].str.lower() == "death"]
            .groupby("Bucket")["Event"]
            .count()
            .cumsum()
            .reset_index(name="Cumulative_Deaths")
        )

        stats = pd.merge(kills_per_bucket, deaths_per_bucket, on="Bucket", how="outer")
        stats[["Cumulative_Kills", "Cumulative_Deaths"]] =\
        stats[["Cumulative_Kills", "Cumulative_Deaths"]].ffill().fillna(0)
        stats["Minutes"] = (stats["Bucket"] + 1) * (bucket_seconds / 60)
        stats["Time_HHMMSS"] = pd.to_timedelta(stats["Minutes"], unit='m').astype(str)
        stats["Time_HHMMSS"] = stats["Time_HHMMSS"].str.replace("0 days ", "")

        self.axes[self.current_axis].set_facecolor(color_pallete["chart_color"])

        line_kills, = self.axes[self.current_axis].plot(
            stats["Time_HHMMSS"],
            stats["Cumulative_Kills"],
            label="Kills",
            marker='o',
            color="#008000"
        )

        # Add data labels
        for x, y in zip(line_kills.get_xdata(), line_kills.get_ydata()):
            self.axes[self.current_axis].annotate(
                f"{int(y)}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),  # slightly above the point
                ha='center',
                color=color_pallete["font_color"]
            )

        line_deaths, = self.axes[self.current_axis].plot(stats["Time_HHMMSS"], stats["Cumulative_Deaths"],
                 label="Deaths", marker='x', color="#ff0000")
        
        # Add data labels
        for x, y in zip(line_deaths.get_xdata(), line_deaths.get_ydata()):
            self.axes[self.current_axis].annotate(
                f"{int(y)}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),  # slightly above the point
                ha='center',
                color=color_pallete["font_color"]
            )
        
        tick_minutes = [b for b in stats["Time_HHMMSS"]]
        self.axes[self.current_axis].set_xticklabels(tick_minutes, color=color_pallete["font_color"], rotation=65)
        self.axes[self.current_axis].tick_params(axis='y', colors=color_pallete["font_color"])
        self.axes[self.current_axis].set_title(f"Cumulative Kills and Deaths (Buckets of {bucket_seconds//60} min)",
                  color=color_pallete["font_color"],
                  loc='left')
        self.axes[self.current_axis].legend()
        self.axes[self.current_axis].grid(True)

        self.current_axis+=1


    def kd_ratio_timeline_chart(self, color_pallete: dict, width, height):
        df = pd.read_csv(self.csv_file)

        bucket_seconds = self.bucket_len * 60
        df["Seconds"] = pd.to_timedelta(df["Timestamp"]).dt.total_seconds()
        df = df.sort_values("Seconds")
        df["Bucket"] = (df["Seconds"] // bucket_seconds).astype(int)

        df.to_csv(f"{self.output_dir}/timestamp_sorted.csv", index=False)

        kills_per_bucket = (
            df[df["Event"].str.lower() == "kill"]
            .groupby("Bucket")["Event"]
            .count()
            .cumsum()
            .reset_index(name="Cumulative_Kills")
        )

        deaths_per_bucket = (
            df[df["Event"].str.lower() == "death"]
            .groupby("Bucket")["Event"]
            .count()
            .cumsum()
            .reset_index(name="Cumulative_Deaths")
        )

        stats = pd.merge(
            kills_per_bucket,
            deaths_per_bucket,
            on="Bucket",
            how="outer"
        )

        stats[["Cumulative_Kills", "Cumulative_Deaths"]] = (
            stats[["Cumulative_Kills", "Cumulative_Deaths"]]
            .ffill()
            .fillna(0)
        )

        stats["KD_Ratio"] = np.where(
            stats["Cumulative_Deaths"] > 0,
            stats["Cumulative_Kills"] / stats["Cumulative_Deaths"],
            stats["Cumulative_Kills"]
        )

        stats["KD_Ratio"] = stats["KD_Ratio"].round(2)
        stats["Minutes"] = (stats["Bucket"] + 1) * (bucket_seconds / 60)

        stats["Time_HHMMSS"] = (
            pd.to_timedelta(stats["Minutes"], unit='m')
            .astype(str)
            .str.replace("0 days ", "", regex=False)
        )

        self.axes[self.current_axis].set_facecolor(
            color_pallete["chart_color"]
        )

        line_kdr, = self.axes[self.current_axis].plot(
            stats["Time_HHMMSS"],
            stats["KD_Ratio"],
            label="K/D Ratio",
            marker='o',
            linewidth=2,
            color=color_pallete['elements_color']
        )

        for x, y in zip(
            line_kdr.get_xdata(),
            line_kdr.get_ydata()
        ):
            self.axes[self.current_axis].annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),
                ha='center',
                color=color_pallete["font_color"]
            )

        self.axes[self.current_axis].axhline(
            y=1,
            color='gray',
            linestyle='--',
            alpha=0.6
        )

        tick_minutes = [b for b in stats["Time_HHMMSS"]]

        self.axes[self.current_axis].set_xticklabels(
            tick_minutes,
            color=color_pallete["font_color"],
            rotation=65
        )

        self.axes[self.current_axis].tick_params(
            axis='y',
            colors=color_pallete["font_color"]
        )

        self.axes[self.current_axis].set_ylabel(
            "K/D Ratio",
            color=color_pallete["font_color"]
        )

        self.axes[self.current_axis].set_title(
            f"K/D Ratio Over Time (Buckets of {bucket_seconds // 60} min)",
            color=color_pallete["font_color"],
            loc='left'
        )

        self.axes[self.current_axis].legend()
        self.axes[self.current_axis].grid(True)

        self.current_axis += 1


    def save_report(self, report_num: int):
        self.fig.savefig(f"{self.output_dir}/summary_report{report_num}.png", dpi=300, bbox_inches="tight")
