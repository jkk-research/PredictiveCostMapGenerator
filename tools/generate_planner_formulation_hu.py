from pathlib import Path
import json
import zipfile

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from latex2mathml.converter import convert as latex_to_mathml
from mathml2omml import convert as mathml_to_omml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'doc' / 'pcm_longitudinal_matematikai_leiras_hu.docx'
document = Document()
section = document.sections[0]
section.page_width = Cm(21)
section.page_height = Cm(29.7)
section.top_margin = Cm(1.55)
section.bottom_margin = Cm(1.55)
section.left_margin = Cm(1.7)
section.right_margin = Cm(1.7)
section.header_distance = Cm(0.65)
section.footer_distance = Cm(0.65)
normal = document.styles['Normal']
normal.font.name = 'Cambria'
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(4)
normal.paragraph_format.line_spacing = 1.02
normal.paragraph_format.widow_control = True
for name in ('Heading 1', 'Heading 2'):
    style = document.styles[name]
    style.font.name = 'Cambria'
    style.font.size = Pt(12)
    style.font.bold = True
    style.font.color.rgb = RGBColor.from_string('173A40')
    style.paragraph_format.space_before = Pt(7)
    style.paragraph_format.space_after = Pt(4)
    style.paragraph_format.keep_with_next = True

header = section.header.paragraphs[0]
header.text = 'PREDIKTÍV HOSSZIRÁNYÚ TERVEZÉS  |  MATEMATIKAI LEÍRÁS'
header.runs[0].font.size = Pt(8)
header.runs[0].font.color.rgb = RGBColor.from_string('555555')
footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
footer.add_run('2026. november 18.  •  ').font.size = Pt(8)
field = OxmlElement('w:fldSimple')
field.set(qn('w:instr'), 'PAGE')
footer._p.append(field)

equation_count = 0
math_object_count = 0


def paragraph(text, small=False):
    result = document.add_paragraph(text)
    result.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if small:
        for run in result.runs:
            run.font.size = Pt(8.5)
    return result


def heading(text, new_page=False):
    result = document.add_paragraph(text, 'Heading 1')
    result.paragraph_format.page_break_before = new_page
    return result


def equation(latex):
    global equation_count, math_object_count
    equation_count += 1
    table = document.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Cm(16.5)
    table.columns[1].width = Cm(1.1)
    table.cell(0, 0).width = Cm(16.5)
    table.cell(0, 1).width = Cm(1.1)
    for cell in table.rows[0].cells:
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        properties = cell._tc.get_or_add_tcPr()
        margins = OxmlElement('w:tcMar')
        for side in ('top', 'left', 'bottom', 'right'):
            value = OxmlElement('w:' + side)
            value.set(qn('w:w'), '0')
            value.set(qn('w:type'), 'dxa')
            margins.append(value)
        properties.append(margins)
    row_properties = table.rows[0]._tr.get_or_add_trPr()
    row_properties.append(OxmlElement('w:cantSplit'))
    expressions = (latex,) if isinstance(latex, str) else latex
    for index, source in enumerate(expressions):
        expression = (table.cell(0, 0).paragraphs[0] if index == 0
                      else table.cell(0, 0).add_paragraph())
        expression.alignment = WD_ALIGN_PARAGRAPH.CENTER
        expression.paragraph_format.space_before = Pt(2)
        expression.paragraph_format.space_after = Pt(4)
        omml = mathml_to_omml(latex_to_mathml(source))
        wrapper = parse_xml('<root xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
                            + omml + '</root>')
        expression._p.append(wrapper[0])
        math_object_count += 1
    number = table.cell(0, 1).paragraphs[0]
    number.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    number.paragraph_format.space_after = Pt(4)
    number.add_run('({})'.format(equation_count)).font.size = Pt(9)


title = document.add_paragraph()
title.paragraph_format.space_after = Pt(5)
run = title.add_run('GP-költségmezőn alapuló\nprediktív hosszirányú tervező')
run.font.name = 'Cambria'
run.font.size = Pt(17)
run.bold = True
paragraph('A /scenario bemenettől a sebességparancsig, a GP-átlag globális közelítésével')

heading('1. Bemenetek, jelölések és a költségmező felépítése')
paragraph('A pcm_longitudinal tervező az előírt, rendezett célút mentén optimalizálja a sebességet; '
          'nem végez önálló oldalirányú pályakeresést. A /scenario üzenet első útvonala adja a '
          'célgeometriát és a megengedett célsebességet, a további útvonalak pontjai egyetlen '
          'egyesített járhatófelület-halmazt alkotnak. A local_moving_objects mezőből két '
          'objektumcsoport készül: P a gyalogosoké, V az összes többi elfogadott osztályé '
          '(így a kerékpárosoké is). A zérus sebességű objektum is ebbe a feldolgozásba kerül; '
          'a geometriai GP-rétegek nem álló akadályokat jelentenek.')
paragraph('Az aktuális állapotot az /ego és – az aktuátoros modellnél – '
          'a /simulator/actuator_state szolgáltatja. Legyen T a horizont, N a költségmező '
          'kiértékelési pontjainak száma, o az objektum, b az előrejelzési ág indexe. '
          'Az ágakat a tervező a közös időrácsra interpolálja; előrejelzés hiányában '
          'állandó sebességű mozgást feltételez.')
equation(r't_j=jT/N,\quad z_i=(x_{ob}(t_j),y_{ob}(t_j),t_j)^{\mathsf T},\quad i=(o,b,j).')
paragraph('A csoportonként külön illesztett, anizotrop Gauss-kernelben q és z tér–időbeli '
          'pontok, σ a kernel amplitúdója, ℓₓ, ℓᵧ és ℓₜ pedig a korrelációs hosszskálák. '
          'A paraméterek előre rögzítettek, nem az optimalizálás döntési változói.')
equation(r'k(q,z)=\sigma^2\exp\!\left[-\frac12\left(\frac{(q_x-z_x)^2}{\ell_x^2}'
         r'+\frac{(q_y-z_y)^2}{\ell_y^2}+\frac{(q_t-z_t)^2}{\ell_t^2}\right)\right].')
paragraph('Legyen pᵢ az objektum létezési valószínűségének és az ág konfidenciájának '
          'szorzata, c_d a d ∈ {P,V} csoport súlyossági költsége. A célérték és a diagonális '
          'zajvariancia időfüggő és bizonytalanságfüggő:')
equation(r'p_i=e_o\gamma_{ob},\qquad y_i=1+p_i c_d,\qquad '
         r'\nu_i=\nu_{0,d}+\nu_{1,d}t_j+p_i(1-p_i)c_d^2.')
paragraph('Az M aktív támaszponthoz tartozó K kernelmátrix és α együtthatóvektor '
          'határozza meg az eredeti objektummezőt. A −1 eltolás az objektumoktól távoli '
          'alapszintet adja; a mező értéke költség, nem ütközési valószínűség.')
equation((r'K_{ij}=k(z_i,z_j),\qquad \alpha=(K+\operatorname{diag}(\nu_i))^{-1}y,',
          r'\mu_d(q)=-1+\sum_{i=1}^{M}\alpha_i k(q,z_i).'))
paragraph('A célút L és a járható felület D külön, kétdimenziós GP-t kap: a kernelből '
          'az időtag elmarad, az illesztés állandó negatív célértékekkel történik. '
          'Itt Z_r a geometriai támaszponthalmaz, c_r annak költségszintje, σₙ,ᵣ a zaj szórása.')
equation(r'\mu_r(\xi)=k_r(\xi,Z_r)^{\mathsf T}'
         r'(K_r+\sigma_{n,r}^{2}I)^{-1}c_r\mathbf 1,\qquad r\in\{L,D\}.')
paragraph('Jelenlegi indítási példa: T = 5 s, N = 10; legfeljebb 5–5 objektum, '
          'objektumonként 2 ág, tehát összesen legfeljebb 200 aktív objektumtámaszpont. '
          'A geometriai GP-k külön-külön 128 helyet tartanak fenn. A névleges ciklusidő '
          '50 ms, ami nem végrehajtási időgarancia.', small=True)

heading('2. A GP-átlag globális közelítése és a közös költség', new_page=True)
paragraph('A global_mean mód a két objektumcsoportban egymástól függetlenül, de az '
          'összes megtartott objektum és ág pontjait közösen kezeli. A kiválasztás '
          'dimenziótlan, kernelhosszskálákkal normált távolságot használ. Az S indexhalmaz '
          'kezdeti elemszáma min(M, max(m_min, ⌈ρM⌉)); első pontja a legnagyobb abszolút '
          'eredeti átlagértékű pont, a további kezdőpontok legtávolabbi pont kiválasztásával adódnak:')
equation(r'd^2(z_i,z_j)=\sum_{a\in\{x,y,t\}}\frac{(z_{i,a}-z_{j,a})^2}{\ell_a^2},'
         r'\qquad i^*=\operatorname{arg\,max}_{i\notin S}\min_{j\in S}d^2(z_i,z_j).')
paragraph('A kiválasztott pontokhoz nem az eredeti α-együtthatók részhalmaza tartozik: '
          'a módszer a teljes eredeti átlagot vetíti a kiválasztott kernelbázisra. '
          'Az S sor- és oszlopkiválasztást jelöl, λ = σ²·10⁻¹⁰ numerikus regularizálás:')
equation(r'\beta=(K_{SS}+\lambda I)^{-1}K_{S,:}\alpha,\qquad '
         r'\mu_d^S(q)=-1+\sum_{i\in S}\beta_i k(q,z_i).')
paragraph('Legyen E_S a csökkentett vektort az eredeti indextérbe beágyazó mátrix. '
          'Az RBF reprodukáló magú Hilbert-terében számított maradéknorma minden '
          'lekérdezési pontra abszolút átlaghibakorlátot ad. Az η ≥ 0 tag a kódban '
          'alkalmazott lebegőpontos kerekítési védőtag:')
equation(r'r=\alpha-E_S\beta,\qquad \varepsilon_d=\sigma\sqrt{\max(0,r^{\mathsf T}Kr)+\eta},'
         r'\qquad |\mu_d(q)-\mu_d^S(q)|\leq\varepsilon_d.')
paragraph('A kezdőhalmaz után az algoritmus maradékhiba alapján bővíti S-t, előnyben '
          'részesítve a lefedési sugaron kívüli pontokat. A leálláshoz a globális '
          'hibatűrésnek és a távolsági lefedésnek is teljesülnie kell; ezért ez a mód '
          'nem rögzített pontkorlátú. Jelenleg m_min = 10, ρ = 0,2, a lefedési sugár 3, '
          'a hibatűrés 0,02. Elfogadhatatlan közelítési hiba vagy túl nagy hibamargóból '
          'adódó költség-alapszint esetén az eredeti teljes átlag tér vissza. A teljes '
          'szimbolikus kapacitás így megmarad: az aktív pontok csökkenése önmagában '
          'nem garantál gyorsabb optimalizálást.')
paragraph('A tervező a közelített objektummezőhöz hozzáadja a hibakorlátot, így az '
          'adott eredeti GP-átlagot felülről becsli. A két objektummezőt simított maximum, '
          'a két geometriai mezőt minimum kapcsolja össze; a végső C költség:')
equation(r'\mu_d^+=\mu_d^S+\varepsilon_d,\qquad '
         r'H(a,b)=\frac{a+b+\sqrt{(a-b)^2+0.01}}{2}.')
equation(r'C(q)=H\!\left(H(\mu_P^+(q),\mu_V^+(q)),'
         r'\min(\mu_L(q_{xy}),\mu_D(q_{xy}))\right).')
paragraph('Eltérő lehetőség a gp_hard_reduction: a célúttól távoli objektumok pontjai '
          'kiesnek a jelöltek közül, majd a távolságalapú kiválasztás rögzített keretig '
          'tart. A megtartott pontok eredeti együtthatói változatlanok:')
equation(r'\mu_d^{\mathrm{hard}}(q)=-1+\sum_{i\in S}\alpha_i k(q,z_i),'
         r'\qquad \varepsilon_d^{\mathrm{applied}}=0.')
paragraph('Ebben az alternatívában nincs projekció és nincs tanúsított hibakorlát; '
          'a nulla kizárólag az alkalmazott hibamargó hiányát jelenti. A mező torzulhat, '
          'az akadályköltség csökkenhet vagy nőhet. A pontkeret fele a gyalogos-, fele '
          'a másik objektum-GP-é; páratlan keretnél az előbbi kapja a többlethelyet. '
          'A két geometriai GP egyik egyszerűsítési módban sem változik.', small=True)

heading('3. Predikció, optimalizálás és a vezérlőparancs', new_page=True)
paragraph('A jelenlegi dynamic_actuators változat négy döntési változója '
          'c = (c₁,c₂,c₃,c₄)ᵀ. A v₀ mért kezdeti sebességhez rögzített, normált időben '
          'negyedfokú polinom a kért sebességet paraméterezi, nem a tényleges járműsebességet:')
equation(r'v_c(t;c)=v_0+\sum_{n=1}^{4}c_n(t/T)^n,\qquad 0\leq t\leq T.')
paragraph('A χ állapotvektor a jármű helyzetét, irányát, sebességkomponenseit, '
          'perdülési sebességét, az aktuátorállapotokat, az energiákat és az út menti '
          'előrehaladást tartalmazza. Az F_h diszkrét jármű–hajtáslánc-modell magában '
          'foglalja a P-sebességszabályozást, a szűrőt, nyomatékkorlátokat, '
          'ellenállásokat és az útemelkedést. Belső fizikai egyenleteit itt nem bontjuk ki. '
          'A célúthoz kötött kormányzást δ_k, az emelkedést θ_k jelöli:')
equation(r'\chi_{k+1}=F_h(\chi_k,v_c(\min(t_k+t_c,T);c),\delta_k,\theta_k),'
         r'\qquad t_c=0.25\ \mathrm{s}.')
paragraph('A modell az aktuális ego-koordinátarendszerben számol. A GP-lekérdezés '
          'előtt a pozíciót a /scenario időbélyegéhez tartozó rendszerbe transzformálja: '
          'qⱼ = ((Rψ ξⱼ + d)ᵀ, tⱼ + Δt)ᵀ; Rψ és d a két rendszer relatív forgatása '
          'és eltolása, Δt a jelenet kora. Jelölje Cⱼ = C(qⱼ), uⱼ a prediktált tényleges '
          'hosszirányú sebességet, aₓ,ⱼ és aᵧ,ⱼ a tényleges gyorsulásokat, '
          'v_ref a célsebességet. Az optimalizálás:')
equation((r'c^*=\operatorname{arg\,min}_c J(c),',
          r'J=w_CJ_C+w_vJ_v+w_xJ_x+w_yJ_y+w_BJ_B+w_E\frac{E_{\mathrm{net}}}{E_{\mathrm{ref}}}.'))
equation((r'J_C=\frac1N\sum_j(C_j+1)^2,\qquad '
          r'J_v=\frac{\sum_j(u_j-v_{\mathrm{ref}})^2}{N\max(v_{\mathrm{ref}}^2,1)},',
          r'J_x=\frac{\sum_j a_{x,j}^2}{Na_{x,\max}},\qquad '
          r'J_y=\frac{\sum_j a_{y,j}^2}{Na_{y,\max}^2}.'))
equation((r'J_B=\sum_j\!\left[\exp\!\left(5\,\operatorname{sp}_{10}(C_j)\right)-1\right],',
          r'\operatorname{sp}_{10}(x)=\frac{\ln(1+e^{10x})}{10},\qquad E_{\mathrm{net}}=E_+-E_-.'))
paragraph('Az E₊ a felvett, E₋ a visszatáplált energia. A nemnegatív w súlyok a tagok '
          'fontosságát adják; E_ref = 100 000 J. A Jₓ normalizálása szándékosan a jelenlegi '
          'kódot követi, nem négyzetes aₓ,max nevezőt használ; aktuális súlya nulla. '
          'A nemlineáris korlátok a kiértékelési pontokban:')
equation((r'C_j\leq C_{\max},\qquad |a_{x,j}|\leq a_{x,\max},\qquad |a_{y,j}|\leq a_{y,\max},',
          r'0\leq u_j\leq\max(v_0,v_{\mathrm{ref}}),\qquad s_j\leq s_{\mathrm{end}}.'))
paragraph('Jelenleg C_max = 0,05, aₓ,max = 5 m/s², aᵧ,max = 3 m/s². A belső '
          'parancsminták felső sebességkorlátot is kapnak. Az SLSQP CasADi-deriváltakat '
          'használ; sikertelenség esetén ugyanazon feladat IPOPT-újrapróbálása következhet. '
          'Csak sikeres, véges és a megengedett numerikus korlátsértésen belüli eredmény '
          'után kerül ki új parancs a /control/command/control_cmd témára:')
equation(r'v_{\mathrm{cmd}}=\max(0,v_c(t_c;c^*)).')
paragraph('Elutasított megoldásból nincs új parancs vagy generált fékezési helyettesítő pálya. '
          'A következő ciklus új mérésekkel tervez; a mintavételi korlátok önmagukban nem '
          'bizonyítanak folytonos idejű ütközésmentességet. A szimulátor opcionális PID/grade '
          'kiegészítéseit a tervező P-alapú aktuátormodellje nem követi automatikusan.', small=True)

heading('4. Hatásfok-LUT és annak optimalizálási célú egyszerűsítése', new_page=True)
paragraph('A dynamic_actuators modell a hajtáslánc dinamikáját továbbra is időben '
          'léptetett, állapotos modellként kezeli: a motor fordulatszáma, a tengely '
          'csavarodása, a motor nyomatéka és annak változási korlátja megmarad. A '
          'hatásfoktérkép csupán azt határozza meg, hogyan lesz a mechanikai '
          'teljesítményből villamos energia. A CSV két külön táblát tartalmaz: '
          'motorüzemre és regenerációra. A lekérdezési fordulatszámot abszolút '
          'motorfordulatból rpm-re, a nyomatékot abszolútértékre alakítja; mindkét '
          'tengelyt a táblázat tartományára korlátozza.')
equation(r'\omega_{\mathrm{rpm}}=\operatorname{clip}\!\left(\frac{60|\omega_m|}{2\pi},'
         r'\omega_{\min},\omega_{\max}\right),\qquad '
         r'\tau=\operatorname{clip}(|\tau_m|,\tau_{\min},\tau_{\max}).')
paragraph('Az adott rácscellában legyen a és b a fordulatszám-, illetve nyomatékirányú '
          'interpolációs arány. Az ηᵐ és ηʳ értékeket a megfelelő üzemmód táblájának '
          'négy szomszédos csomópontjából bilineáris interpoláció adja; a jelölés '
          'm = motorüzem, r = regeneráció:')
equation(r'\eta^{m/r}(\omega,\tau)=(1-a)(1-b)\eta_{ij}^{m/r}'
         r'+a(1-b)\eta_{i+1,j}^{m/r}+(1-a)b\eta_{i,j+1}^{m/r}'
         r'+ab\eta_{i+1,j+1}^{m/r}.')
paragraph('A mechanikai teljesítmény előjele választja ki az üzemmódot. Hajtáskor a '
          'veszteségek miatt a villamos teljesítmény nagyobb a mechanikainál; '
          'regenerációnál a visszanyert teljesítmény annak η-szorosa. A külön gyűjtött '
          'energia pozitív mennyiség, a célfüggvényben szereplő nettó energia viszont '
          'előjeles:')
equation((r'P_m=\tau_m\omega_m,\qquad '
          r'P_e=\begin{cases}P_m/\eta^m,&P_m\geq0,\\P_m\eta^r,&P_m<0,\end{cases}',
          r'E_+=\sum_k\max(P_{e,k},0)\Delta t_k,\qquad '
          r'E_-=\sum_k\max(-P_{e,k},0)\Delta t_k,\qquad E_{\mathrm{net}}=E_+-E_-.'))
paragraph('LUT módban a tervező közvetlenül a szimulátorral azonos CSV-t és bilineáris '
          'interpolációt használ; ez a rácscellákon belül az eredeti térképet követi, '
          'de a cellahatárokon a derivált változhat. Az aktuális launch a spline '
          'reprezentációt választja. Ekkor a bilineárisan kiértékelt térképet sűrű '
          'fordulatszám–nyomaték rácson mintavételezzük, majd a két tengely rácscsomópontjain '
          'csonkolt hatványú köbös regressziós spline-bázist illesztünk legkisebb négyzetekkel.')
equation(r'\eta(\omega,\tau)\approx\sum_{r=1}^{R}f_r(\omega)g_r(\tau),'
         r'\qquad f_r\in\mathcal B_\omega,\quad g_r\in\mathcal B_\tau.')
paragraph('A szeparálható alacsony rangú alakot a spline-illesztés ortogonalizált '
          'mátrixain végzett szingulárisérték-felbontással kapjuk. A legkisebb olyan '
          'rangot tartjuk meg, amelynek maximális eltérése közel a teljes spline '
          'illesztési hibájánál marad; ez csökkenti a szimbolikus kiértékelés költségét '
          'és megőrzi a sima deriváltakat. Az alternatív polynomial mód teljes '
          'fokszámkorlátos kétdimenziós polinomot illeszt ugyanerre a sűrű mintára. '
          'Mindkét közelítésre maximális abszolút hibaküszöb érvényes; ha a hiba túllépi '
          'a konfigurált 0,05 értéket, vagy az illesztett hatásfok kiesik az '
          'érvényes tartományból, az inicializálás hibával leáll, és a LUT mód használatát jelzi.')
equation(r'\max_{(\omega,\tau)\in\mathcal G}|\eta_{\mathrm{fit}}-\eta_{\mathrm{LUT}}|'
         r'\leq\epsilon_\eta=0.05,\qquad 0.05<\eta_{\mathrm{fit}}\leq1.')
paragraph('Az egyszerűsítés tehát a LUT folytonos, differenciálható közelítésére és '
          'annak tömör kiértékelésére vonatkozik; nem a nyomaték-, energia- vagy '
          'járműdinamika elhagyására. Az illesztési hiba a sűrű ellenőrzőrácsra '
          'vonatkozó numerikus feltétel, nem minden folytonos ponton igazolt '
          'felső korlát. A szállított hatásfoktérkép szemléltető jellegű, nem mért '
          'járműadat. Forrás: pcm_powertrain.py és pcm_longitudinal.launch.py.', small=True)

paragraph('Forrásalap: pcm_scenario.py; pcm_kernels.py; pcm_longitudinal.py; '
          'pcm_longitudinal_dynamics.py; pcm_powertrain.py; pcm_longitudinal.launch.py. '
          'Állapot: 2026. 11. 18.; módszerleírás, nem biztonsági tanúsítás.', small=True)

document.core_properties.title = 'GP-költségmezőn alapuló prediktív hosszirányú tervező'
document.core_properties.subject = 'Magyar matematikai leírás a global_mean GP-közelítéssel'
document.core_properties.author = 'CRP – műszaki dokumentáció'
document.core_properties.language = 'hu-HU'
language = normal.element.get_or_add_rPr()
lang = OxmlElement('w:lang')
lang.set(qn('w:val'), 'hu-HU')
language.append(lang)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
document.save(str(OUTPUT))
with zipfile.ZipFile(str(OUTPUT)) as archive:
    xml = archive.read('word/document.xml')
    assert b'hard' in xml and b'global_mean' in xml
    assert xml.count(b'<m:oMath>') == math_object_count
print(json.dumps({'output': str(OUTPUT), 'editable_equations': equation_count,
                  'math_objects': math_object_count,
                  'intended_pages': 4, 'bytes': OUTPUT.stat().st_size}, ensure_ascii=False))