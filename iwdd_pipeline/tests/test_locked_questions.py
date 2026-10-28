"""Blocca il set di domande: il file di configurazione deve contenere ESATTAMENTE le 24 domande scelte dall'utente (testo italiano
identico, salvo il refuso 'un un' corretto) e le 11 suggerite, con la traduzione inglese concordata; e il prompt che arriva a
Visual-JEV deve contenere quel testo. Se cambi una domanda in config/questions.json questo test fallisce di proposito:
aggiorna anche la lista qui sotto e usa un nuovo --run (le feature gia' estratte non sono piu' valide)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iwdd.questions import load_questions, render_prompts   # noqa: E402

# (id, italiano ESATTO come scritto dall'utente, inglese dato al modello)
UTENTE = [
    ("carry_waste", "Una persona sta trasportando un rifiuto?", "Is a person carrying a piece of waste?"),
    ("waste_put_down", "Un rifiuto viene appoggiato a terra?", "Is a piece of waste being put down on the ground?"),
    ("waste_untouched", "C'è un un rifiuto a terra che nessuno sta toccando?", "Is there a piece of waste on the ground that nobody is touching?"),
    ("open_bag", "Una persona tiene un sacco aperto e ci mette dentro degli oggetti?", "Is a person holding an open bag and putting items into it?"),
    ("gloves", "Una persona indossa guanti per raccogliere rifiuti?", "Is a person wearing gloves for picking up waste?"),
    ("trunk_open", "Un veicolo è fermo con baule, portellone o portiera aperti?", "Is a vehicle stopped with its trunk, tailgate or door open?"),
    ("air_object", "Un oggetto è in aria o sta cadendo vicino a un veicolo o a una persona?", "Is an object in the air or falling close to a vehicle or a person?"),
    ("vehicle_by_pile", "Un veicolo è fermo accanto a un mucchio di sacchi di rifiuti?", "Is a vehicle stopped next to a pile of waste bags?"),
    ("pile_on_ground", "C'è un mucchio di sacchi o rifiuti a terra?", "Is there a pile of bags or waste on the ground?"),
    ("waste_by_bin", "Un rifiuto viene appoggiato a terra accanto a un bidone o cassonetto?", "Is a piece of waste being put on the ground next to a bin or dumpster?"),
    ("waste_into_bin", "Una persona mette un rifiuto dentro un bidone o cassonetto?", "Is a person putting a piece of waste inside a bin or dumpster?"),
    ("bulky_moving", "Una persona trasporta o sposta un oggetto ingombrante (materasso, mobile, elettrodomestico)?", "Is a person carrying or moving a bulky object (mattress, furniture, appliance)?"),
    ("walking_away", "Una persona si allontana da un oggetto che resta a terra?", "Is a person walking away from an object that stays on the ground?"),
    ("vehicle_trips", "Una persona fa più viaggi fra un veicolo e un mucchio di oggetti?", "Is a person making several trips between a vehicle and a pile of objects?"),
    ("only_passing", "Passano solo persone o veicoli, senza che nessun oggetto venga maneggiato?", "Are there only people or vehicles passing, with no object being handled?"),
    ("c_put_leave", "Una persona appoggia una borsa o un oggetto a terra e si allontana.", "A person puts a bag or an object on the ground and walks away from it."),
    ("c_throw_vehicle", "Una persona lancia o lascia cadere un oggetto da un veicolo.", "A person throws or drops an object from a vehicle."),
    ("c_unload", "Una persona scarica oggetti da un veicolo e li lascia a terra.", "A person unloads objects from a vehicle and leaves them on the ground."),
    ("c_collect", "Alcune persone raccolgono rifiuti da terra e li mettono in un sacco.", "People are picking up litter from the ground and putting it into a bag."),
    ("c_drop_walking", "Una persona lascia cadere o getta un oggetto mentre cammina.", "A person drops or tosses an object while walking."),
    ("c_waste_in_bin", "Una persona mette dei rifiuti dentro un bidone o un cassonetto.", "A person puts waste inside a bin or dumpster."),
    ("c_pickup_carry", "Una persona raccoglie un oggetto da terra e lo porta via.", "A person picks up an object from the ground and carries it away."),
    ("c_leave_bulky", "Una persona lascia un oggetto ingombrante sul marciapiede.", "A person leaves a bulky item on the pavement."),
    ("scene_summary", "Quale affermazione descrive meglio cosa succede agli oggetti nella scena?", "Which statement best describes what happens with the objects in the scene?"),
]
SUGGERITE = [
    ("garbage_truck", "C'è un camion della raccolta rifiuti, o un operatore che svuota un bidone in un camion?", "Is there a garbage collection truck, or a worker emptying a bin into a truck?"),
    ("hands_on_object", "Una persona tocca o maneggia un oggetto con le mani?", "Is a person touching or handling an object with their hands?"),
    ("person_visible", "C'è almeno una persona visibile nei frame?", "Is at least one person visible in the frames?"),
    ("near_waste_idle", "Una persona è vicina ai rifiuti senza toccarli?", "Is a person standing close to waste without touching it?"),
    ("hold_first", "Nei primi frame (riga in alto) una persona tiene in mano una borsa o un oggetto?", "In the first frames (top row), is a person holding a bag or an object?"),
    ("hold_last", "Negli ultimi frame (riga in basso) una persona tiene in mano una borsa o un oggetto?", "In the last frames (bottom row), is a person holding a bag or an object?"),
    ("ground_first", "Nei primi frame (riga in alto) c'è una borsa o un oggetto a terra?", "In the first frames (top row), is there a bag or an object on the ground?"),
    ("ground_last", "Negli ultimi frame (riga in basso) c'è una borsa o un oggetto a terra che nessuno sta toccando?", "In the last frames (bottom row), is there a bag or an object on the ground with nobody touching it?"),
    ("heavy_material", "Una persona scarica o trasporta pneumatici, macerie o materiale edile?", "Is a person unloading or carrying tires, rubble or construction material?"),
    ("small_fall", "Un piccolo oggetto cade a terra da una persona o da un veicolo?", "Does a small object fall from a person or a vehicle to the ground?"),
    ("arm_out_vehicle", "Una mano o un braccio escono dal finestrino o dalla portiera di un veicolo?", "Is a hand or an arm extended out of a vehicle window or door?"),
]
MC_EN = ["A person leaves objects on the ground and walks away", "A person throws or unloads objects from a vehicle",
         "People pick up litter or collect objects", "Nothing is being left, thrown or collected"]


class TestSetBloccato(unittest.TestCase):
    def setUp(self):
        self.qs = {q.id: q for q in load_questions()}

    def test_ci_sono_tutte_e_solo_queste(self):
        self.assertEqual(set(self.qs), {i for i, _, _ in UTENTE + SUGGERITE})
        self.assertEqual(len(self.qs), 35)

    def test_italiano_identico_a_quello_dell_utente(self):
        for qid, it_user, _ in UTENTE:
            # unica differenza ammessa: il refuso "un un" nella tua frase 3, corretto in "un"
            self.assertEqual(self.qs[qid].it, it_user.replace("un un", "un"), qid)
        for qid, it, _ in SUGGERITE:
            self.assertEqual(self.qs[qid].it, it, qid)

    def test_inglese_concordato(self):
        for qid, _, en in UTENTE + SUGGERITE:
            self.assertEqual(self.qs[qid].en, en, qid)
        self.assertEqual(list(self.qs["scene_summary"].options), MC_EN)

    def test_origine(self):
        self.assertEqual({i for i, q in self.qs.items() if q.origin == "utente"}, {i for i, _, _ in UTENTE})
        self.assertEqual({i for i, q in self.qs.items() if q.origin == "suggerita"}, {i for i, _, _ in SUGGERITE})


class TestPromptRicevuto(unittest.TestCase):
    """Il prompt costruito con le funzioni originali di Visual-JEV contiene proprio le domande bloccate."""

    def test_prompt(self):
        qs = load_questions()
        pr = dict(render_prompts(qs, "rows"))
        self.assertEqual(len(pr), 35)
        for q in qs:
            p = pr[q.id]
            self.assertIn(f"Question: {q.en}", p, q.id)
            self.assertTrue(p.rstrip().endswith("Answer:"), q.id)
            self.assertIn("<|vision_start|><|image_pad|><|vision_end|>", p)
            self.assertIn("The top row shows the first three frames", p)
            if q.kind == "yesno":
                self.assertIn("A. yes\nB. no", p, q.id)
            elif q.kind == "claim":
                self.assertIn("A. supported by the image\nB. contradicted by the image\nC. not determined by the image", p, q.id)
            else:
                for k, o in enumerate(MC_EN):
                    self.assertIn(f"{chr(65 + k)}. {o}", p)

    def test_senza_contesto(self):
        pr = dict(render_prompts(load_questions(), "none"))
        self.assertNotIn("3x3 grid", pr["carry_waste"])


class TestSoloInglese(unittest.TestCase):
    """A Visual-JEV arriva SOLO inglese: file delle domande, contesto, system prompt, domande e opzioni."""

    def test_file_domande_ascii(self):
        text = (Path(__file__).resolve().parents[1] / "config" / "questions.json").read_text(encoding="utf-8")
        self.assertTrue(text.isascii(), "config/questions.json contiene caratteri non inglesi")

    def test_ogni_prompt_e_ascii(self):
        for mode in ("rows", "basic", "none"):
            for qid, prompt in render_prompts(load_questions(), mode):
                self.assertTrue(prompt.isascii(), f"{qid} ({mode}): prompt non inglese")

    def test_nessuna_parola_italiana_nei_prompt(self):
        parole = (" una ", " un ", " che ", " persona ", " rifiut", " bidone", " del ", " nella ", " sono ")
        for qid, prompt in render_prompts(load_questions(), "rows"):
            low = " " + prompt.lower().replace("\n", " ") + " "
            for w in parole:
                self.assertNotIn(w, low, f"{qid}: parola italiana {w!r}")

    def test_blocca_testo_non_inglese(self):
        import json
        import tempfile
        d = json.loads((Path(__file__).resolve().parents[1] / "config" / "questions.json").read_text(encoding="utf-8"))
        d["questions"][0]["en"] = "Una persona sta trasportando un rifiutoà?"
        f = Path(tempfile.mkdtemp()) / "q.json"
        f.write_text(json.dumps(d), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_questions(f)


if __name__ == "__main__":
    unittest.main()
