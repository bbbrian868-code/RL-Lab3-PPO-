import glob
import os
import shutil

from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.record_writer import RecordWriter


# [說明] 續訓時把 eval_log.csv 中 step > max_step 的列刪掉 (那些評估會在續訓後重做)
def trim_eval_log(csv_path, max_step):
	if not os.path.exists(csv_path):
		return
	with open(csv_path) as f:
		rows = [line for line in f if line.strip()]
	kept = [line for line in rows if int(line.split(",")[0]) <= max_step]
	if len(kept) != len(rows):
		with open(csv_path, "w") as f:
			f.writelines(kept)
		print(f"Trimmed {len(rows) - len(kept)} eval_log rows after step {max_step}")


# [說明] 續訓時用：程序被中止時，TensorBoard 已記錄到比 last.pt 更後面的步數，
# [說明] 續訓會從 last.pt 的步數重跑這一段，TensorBoard 上會出現重疊的折返線。
# [說明] 這裡把舊 event 檔中 step > max_step 的紀錄刪掉 (原檔備份到 tb_backup/，不直接刪除)。
def trim_tb_events(tb_dir, max_step, backup_dir):
	for path in sorted(glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))):
		records = [event_pb2.Event.FromString(r) for r in RawEventFileLoader(path).Load()]
		kept = [e for e in records if e.step <= max_step]
		if len(kept) == len(records):
			continue
		os.makedirs(backup_dir, exist_ok=True)
		shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))
		with open(path, "wb") as f:
			writer = RecordWriter(f)
			for e in kept:
				writer.write(e.SerializeToString())
		print(f"Trimmed {len(records) - len(kept)} TensorBoard records after step {max_step} from {os.path.basename(path)}")
