"""Utilita' per misurare i tempi con tqdm."""
import time
from contextlib import contextmanager

from tqdm.auto import tqdm

def fmt_time(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"

@contextmanager
def timed(desc, results=None):
    """Barra tqdm a un solo passo: a fine blocco mostra quanto tempo ha impiegato (opzionale: lo salva in results)."""
    start = time.perf_counter()
    bar = tqdm(total=1, desc=desc, leave=True)
    try:
        yield bar
    except BaseException:
        bar.close()
        raise
    else:
        bar.update(1)
        bar.close()
    finally:
        if results is not None:
            results[desc] = time.perf_counter() - start
