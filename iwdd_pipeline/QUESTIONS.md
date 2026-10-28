# Domande date a Visual-JEV (35)

Testo inglese = quello esatto passato al modello. Le opzioni le aggiunge il codice di Visual-JEV (`A. yes / B. no`, oppure le opzioni indicate). La traduzione italiana e' solo per leggibilita'.

| # | id | tipo | gruppo | origine | inglese (al modello) | italiano |
|---|---|---|---|---|---|---|
| 1 | `carry_waste` | sì/no | stato | utente | Is a person carrying a piece of waste? | Una persona sta trasportando un rifiuto? |
| 2 | `waste_put_down` | sì/no | azione | utente | Is a piece of waste being put down on the ground? | Un rifiuto viene appoggiato a terra? |
| 3 | `waste_untouched` | sì/no | stato | utente | Is there a piece of waste on the ground that nobody is touching? | C'è un rifiuto a terra che nessuno sta toccando? |
| 4 | `open_bag` | sì/no | raccolta | utente | Is a person holding an open bag and putting items into it? | Una persona tiene un sacco aperto e ci mette dentro degli oggetti? |
| 5 | `gloves` | sì/no | raccolta | utente | Is a person wearing gloves for picking up waste? | Una persona indossa guanti per raccogliere rifiuti? |
| 6 | `trunk_open` | sì/no | veicolo | utente | Is a vehicle stopped with its trunk, tailgate or door open? | Un veicolo è fermo con baule, portellone o portiera aperti? |
| 7 | `air_object` | sì/no | lancio | utente | Is an object in the air or falling close to a vehicle or a person? | Un oggetto è in aria o sta cadendo vicino a un veicolo o a una persona? |
| 8 | `vehicle_by_pile` | sì/no | veicolo | utente | Is a vehicle stopped next to a pile of waste bags? | Un veicolo è fermo accanto a un mucchio di sacchi di rifiuti? |
| 9 | `pile_on_ground` | sì/no | stato | utente | Is there a pile of bags or waste on the ground? | C'è un mucchio di sacchi o rifiuti a terra? |
| 10 | `waste_by_bin` | sì/no | bidone | utente | Is a piece of waste being put on the ground next to a bin or dumpster? | Un rifiuto viene appoggiato a terra accanto a un bidone o cassonetto? |
| 11 | `waste_into_bin` | sì/no | bidone | utente | Is a person putting a piece of waste inside a bin or dumpster? | Una persona mette un rifiuto dentro un bidone o cassonetto? |
| 12 | `bulky_moving` | sì/no | ingombranti | utente | Is a person carrying or moving a bulky object (mattress, furniture, appliance)? | Una persona trasporta o sposta un oggetto ingombrante (materasso, mobile, elettrodomestico)? |
| 13 | `walking_away` | sì/no | azione | utente | Is a person walking away from an object that stays on the ground? | Una persona si allontana da un oggetto che resta a terra? |
| 14 | `vehicle_trips` | sì/no | veicolo | utente | Is a person making several trips between a vehicle and a pile of objects? | Una persona fa più viaggi fra un veicolo e un mucchio di oggetti? |
| 15 | `only_passing` | sì/no | negativi | utente | Are there only people or vehicles passing, with no object being handled? | Passano solo persone o veicoli, senza che nessun oggetto venga maneggiato? |
| 16 | `c_put_leave` | affermazione | affermazione | utente | A person puts a bag or an object on the ground and walks away from it.<br>(supported / contradicted / not determined) | Una persona appoggia una borsa o un oggetto a terra e si allontana. |
| 17 | `c_throw_vehicle` | affermazione | affermazione | utente | A person throws or drops an object from a vehicle.<br>(supported / contradicted / not determined) | Una persona lancia o lascia cadere un oggetto da un veicolo. |
| 18 | `c_unload` | affermazione | affermazione | utente | A person unloads objects from a vehicle and leaves them on the ground.<br>(supported / contradicted / not determined) | Una persona scarica oggetti da un veicolo e li lascia a terra. |
| 19 | `c_collect` | affermazione | affermazione | utente | People are picking up litter from the ground and putting it into a bag.<br>(supported / contradicted / not determined) | Alcune persone raccolgono rifiuti da terra e li mettono in un sacco. |
| 20 | `c_drop_walking` | affermazione | affermazione | utente | A person drops or tosses an object while walking.<br>(supported / contradicted / not determined) | Una persona lascia cadere o getta un oggetto mentre cammina. |
| 21 | `c_waste_in_bin` | affermazione | affermazione | utente | A person puts waste inside a bin or dumpster.<br>(supported / contradicted / not determined) | Una persona mette dei rifiuti dentro un bidone o un cassonetto. |
| 22 | `c_pickup_carry` | affermazione | affermazione | utente | A person picks up an object from the ground and carries it away.<br>(supported / contradicted / not determined) | Una persona raccoglie un oggetto da terra e lo porta via. |
| 23 | `c_leave_bulky` | affermazione | affermazione | utente | A person leaves a bulky item on the pavement.<br>(supported / contradicted / not determined) | Una persona lascia un oggetto ingombrante sul marciapiede. |
| 24 | `scene_summary` | scelta multipla | scena | utente | Which statement best describes what happens with the objects in the scene?<br>A. A person leaves objects on the ground and walks away / Una persona lascia oggetti a terra e si allontana<br>B. A person throws or unloads objects from a vehicle / Una persona lancia o scarica oggetti da un veicolo<br>C. People pick up litter or collect objects / Alcune persone raccolgono rifiuti o oggetti<br>D. Nothing is being left, thrown or collected / Non viene lasciato, lanciato né raccolto nulla | Quale affermazione descrive meglio cosa succede agli oggetti nella scena? |
| 25 | `garbage_truck` | sì/no | negativi | suggerita | Is there a garbage collection truck, or a worker emptying a bin into a truck? | C'è un camion della raccolta rifiuti, o un operatore che svuota un bidone in un camion? |
| 26 | `hands_on_object` | sì/no | negativi | suggerita | Is a person touching or handling an object with their hands? | Una persona tocca o maneggia un oggetto con le mani? |
| 27 | `person_visible` | sì/no | negativi | suggerita | Is at least one person visible in the frames? | C'è almeno una persona visibile nei frame? |
| 28 | `near_waste_idle` | sì/no | negativi | suggerita | Is a person standing close to waste without touching it? | Una persona è vicina ai rifiuti senza toccarli? |
| 29 | `hold_first` | sì/no | direzione | suggerita | In the first frames (top row), is a person holding a bag or an object? | Nei primi frame (riga in alto) una persona tiene in mano una borsa o un oggetto? |
| 30 | `hold_last` | sì/no | direzione | suggerita | In the last frames (bottom row), is a person holding a bag or an object? | Negli ultimi frame (riga in basso) una persona tiene in mano una borsa o un oggetto? |
| 31 | `ground_first` | sì/no | direzione | suggerita | In the first frames (top row), is there a bag or an object on the ground? | Nei primi frame (riga in alto) c'è una borsa o un oggetto a terra? |
| 32 | `ground_last` | sì/no | direzione | suggerita | In the last frames (bottom row), is there a bag or an object on the ground with nobody touching it? | Negli ultimi frame (riga in basso) c'è una borsa o un oggetto a terra che nessuno sta toccando? |
| 33 | `heavy_material` | sì/no | ingombranti | suggerita | Is a person unloading or carrying tires, rubble or construction material? | Una persona scarica o trasporta pneumatici, macerie o materiale edile? |
| 34 | `small_fall` | sì/no | lancio | suggerita | Does a small object fall from a person or a vehicle to the ground? | Un piccolo oggetto cade a terra da una persona o da un veicolo? |
| 35 | `arm_out_vehicle` | sì/no | lancio | suggerita | Is a hand or an arm extended out of a vehicle window or door? | Una mano o un braccio escono dal finestrino o dalla portiera di un veicolo? |
