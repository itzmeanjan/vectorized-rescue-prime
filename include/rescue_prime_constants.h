#pragma once
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

const uint64_t MOD = 18446744069414584321ull;

const cl_ulong MDS[] = {
    2108866337646019936ull % MOD,
    11223275256334781131ull % MOD,
    2318414738826783588ull % MOD,
    11240468238955543594ull % MOD,
    8007389560317667115ull % MOD,
    11080831380224887131ull % MOD,
    3922954383102346493ull % MOD,
    17194066286743901609ull % MOD,
    152620255842323114ull % MOD,
    7203302445933022224ull % MOD,
    17781531460838764471ull % MOD,
    2306881200ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    3368836954250922620ull % MOD,
    5531382716338105518ull % MOD,
    7747104620279034727ull % MOD,
    14164487169476525880ull % MOD,
    4653455932372793639ull % MOD,
    5504123103633670518ull % MOD,
    3376629427948045767ull % MOD,
    1687083899297674997ull % MOD,
    8324288417826065247ull % MOD,
    17651364087632826504ull % MOD,
    15568475755679636039ull % MOD,
    4656488262337620150ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    2560535215714666606ull % MOD,
    10793518538122219186ull % MOD,
    408467828146985886ull % MOD,
    13894393744319723897ull % MOD,
    17856013635663093677ull % MOD,
    14510101432365346218ull % MOD,
    12175743201430386993ull % MOD,
    12012700097100374591ull % MOD,
    976880602086740182ull % MOD,
    3187015135043748111ull % MOD,
    4630899319883688283ull % MOD,
    17674195666610532297ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    10940635879119829731ull % MOD,
    9126204055164541072ull % MOD,
    13441880452578323624ull % MOD,
    13828699194559433302ull % MOD,
    6245685172712904082ull % MOD,
    3117562785727957263ull % MOD,
    17389107632996288753ull % MOD,
    3643151412418457029ull % MOD,
    10484080975961167028ull % MOD,
    4066673631745731889ull % MOD,
    8847974898748751041ull % MOD,
    9548808324754121113ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    15656099696515372126ull % MOD,
    309741777966979967ull % MOD,
    16075523529922094036ull % MOD,
    5384192144218250710ull % MOD,
    15171244241641106028ull % MOD,
    6660319859038124593ull % MOD,
    6595450094003204814ull % MOD,
    15330207556174961057ull % MOD,
    2687301105226976975ull % MOD,
    15907414358067140389ull % MOD,
    2767130804164179683ull % MOD,
    8135839249549115549ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    14687393836444508153ull % MOD,
    8122848807512458890ull % MOD,
    16998154830503301252ull % MOD,
    2904046703764323264ull % MOD,
    11170142989407566484ull % MOD,
    5448553946207765015ull % MOD,
    9766047029091333225ull % MOD,
    3852354853341479440ull % MOD,
    14577128274897891003ull % MOD,
    11994931371916133447ull % MOD,
    8299269445020599466ull % MOD,
    2859592328380146288ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    4920761474064525703ull % MOD,
    13379538658122003618ull % MOD,
    3169184545474588182ull % MOD,
    15753261541491539618ull % MOD,
    622292315133191494ull % MOD,
    14052907820095169428ull % MOD,
    5159844729950547044ull % MOD,
    17439978194716087321ull % MOD,
    9945483003842285313ull % MOD,
    13647273880020281344ull % MOD,
    14750994260825376ull % MOD,
    12575187259316461486ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    3371852905554824605ull % MOD,
    8886257005679683950ull % MOD,
    15677115160380392279ull % MOD,
    13242906482047961505ull % MOD,
    12149996307978507817ull % MOD,
    1427861135554592284ull % MOD,
    4033726302273030373ull % MOD,
    14761176804905342155ull % MOD,
    11465247508084706095ull % MOD,
    12112647677590318112ull % MOD,
    17343938135425110721ull % MOD,
    14654483060427620352ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    5421794552262605237ull % MOD,
    14201164512563303484ull % MOD,
    5290621264363227639ull % MOD,
    1020180205893205576ull % MOD,
    14311345105258400438ull % MOD,
    7828111500457301560ull % MOD,
    9436759291445548340ull % MOD,
    5716067521736967068ull % MOD,
    15357555109169671716ull % MOD,
    4131452666376493252ull % MOD,
    16785275933585465720ull % MOD,
    11180136753375315897ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    10451661389735482801ull % MOD,
    12128852772276583847ull % MOD,
    10630876800354432923ull % MOD,
    6884824371838330777ull % MOD,
    16413552665026570512ull % MOD,
    13637837753341196082ull % MOD,
    2558124068257217718ull % MOD,
    4327919242598628564ull % MOD,
    4236040195908057312ull % MOD,
    2081029262044280559ull % MOD,
    2047510589162918469ull % MOD,
    6835491236529222042ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    5675273097893923172ull % MOD,
    8120839782755215647ull % MOD,
    9856415804450870143ull % MOD,
    1960632704307471239ull % MOD,
    15279057263127523057ull % MOD,
    17999325337309257121ull % MOD,
    72970456904683065ull % MOD,
    8899624805082057509ull % MOD,
    16980481565524365258ull % MOD,
    6412696708929498357ull % MOD,
    13917768671775544479ull % MOD,
    5505378218427096880ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    10318314766641004576ull % MOD,
    17320192463105632563ull % MOD,
    11540812969169097044ull % MOD,
    7270556942018024148ull % MOD,
    4755326086930560682ull % MOD,
    2193604418377108959ull % MOD,
    11681945506511803967ull % MOD,
    8000243866012209465ull % MOD,
    6746478642521594042ull % MOD,
    12096331252283646217ull % MOD,
    13208137848575217268ull % MOD,
    5548519654341606996ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
};

const cl_ulong ARK1[] = {
    13917550007135091859ull % MOD,
    16002276252647722320ull % MOD,
    4729924423368391595ull % MOD,
    10059693067827680263ull % MOD,
    9804807372516189948ull % MOD,
    15666751576116384237ull % MOD,
    10150587679474953119ull % MOD,
    13627942357577414247ull % MOD,
    2323786301545403792ull % MOD,
    615170742765998613ull % MOD,
    8870655212817778103ull % MOD,
    10534167191270683080ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    14572151513649018290ull % MOD,
    9445470642301863087ull % MOD,
    6565801926598404534ull % MOD,
    12667566692985038975ull % MOD,
    7193782419267459720ull % MOD,
    11874811971940314298ull % MOD,
    17906868010477466257ull % MOD,
    1237247437760523561ull % MOD,
    6829882458376718831ull % MOD,
    2140011966759485221ull % MOD,
    1624379354686052121ull % MOD,
    50954653459374206ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    16288075653722020941ull % MOD,
    13294924199301620952ull % MOD,
    13370596140726871456ull % MOD,
    611533288599636281ull % MOD,
    12865221627554828747ull % MOD,
    12269498015480242943ull % MOD,
    8230863118714645896ull % MOD,
    13466591048726906480ull % MOD,
    10176988631229240256ull % MOD,
    14951460136371189405ull % MOD,
    5882405912332577353ull % MOD,
    18125144098115032453ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    6076976409066920174ull % MOD,
    7466617867456719866ull % MOD,
    5509452692963105675ull % MOD,
    14692460717212261752ull % MOD,
    12980373618703329746ull % MOD,
    1361187191725412610ull % MOD,
    6093955025012408881ull % MOD,
    5110883082899748359ull % MOD,
    8578179704817414083ull % MOD,
    9311749071195681469ull % MOD,
    16965242536774914613ull % MOD,
    5747454353875601040ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    13684212076160345083ull % MOD,
    19445754899749561ull % MOD,
    16618768069125744845ull % MOD,
    278225951958825090ull % MOD,
    4997246680116830377ull % MOD,
    782614868534172852ull % MOD,
    16423767594935000044ull % MOD,
    9990984633405879434ull % MOD,
    16757120847103156641ull % MOD,
    2103861168279461168ull % MOD,
    16018697163142305052ull % MOD,
    6479823382130993799ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    13957683526597936825ull % MOD,
    9702819874074407511ull % MOD,
    18357323897135139931ull % MOD,
    3029452444431245019ull % MOD,
    1809322684009991117ull % MOD,
    12459356450895788575ull % MOD,
    11985094908667810946ull % MOD,
    12868806590346066108ull % MOD,
    7872185587893926881ull % MOD,
    10694372443883124306ull % MOD,
    8644995046789277522ull % MOD,
    1422920069067375692ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    17619517835351328008ull % MOD,
    6173683530634627901ull % MOD,
    15061027706054897896ull % MOD,
    4503753322633415655ull % MOD,
    11538516425871008333ull % MOD,
    12777459872202073891ull % MOD,
    17842814708228807409ull % MOD,
    13441695826912633916ull % MOD,
    5950710620243434509ull % MOD,
    17040450522225825296ull % MOD,
    8787650312632423701ull % MOD,
    7431110942091427450ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
};

const cl_ulong ARK2[] = {
    7989257206380839449ull % MOD,
    8639509123020237648ull % MOD,
    6488561830509603695ull % MOD,
    5519169995467998761ull % MOD,
    2972173318556248829ull % MOD,
    14899875358187389787ull % MOD,
    14160104549881494022ull % MOD,
    5969738169680657501ull % MOD,
    5116050734813646528ull % MOD,
    12120002089437618419ull % MOD,
    17404470791907152876ull % MOD,
    2718166276419445724ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    2485377440770793394ull % MOD,
    14358936485713564605ull % MOD,
    3327012975585973824ull % MOD,
    6001912612374303716ull % MOD,
    17419159457659073951ull % MOD,
    11810720562576658327ull % MOD,
    14802512641816370470ull % MOD,
    751963320628219432ull % MOD,
    9410455736958787393ull % MOD,
    16405548341306967018ull % MOD,
    6867376949398252373ull % MOD,
    13982182448213113532ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    10436926105997283389ull % MOD,
    13237521312283579132ull % MOD,
    668335841375552722ull % MOD,
    2385521647573044240ull % MOD,
    3874694023045931809ull % MOD,
    12952434030222726182ull % MOD,
    1972984540857058687ull % MOD,
    14000313505684510403ull % MOD,
    976377933822676506ull % MOD,
    8407002393718726702ull % MOD,
    338785660775650958ull % MOD,
    4208211193539481671ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    2284392243703840734ull % MOD,
    4500504737691218932ull % MOD,
    3976085877224857941ull % MOD,
    2603294837319327956ull % MOD,
    5760259105023371034ull % MOD,
    2911579958858769248ull % MOD,
    18415938932239013434ull % MOD,
    7063156700464743997ull % MOD,
    16626114991069403630ull % MOD,
    163485390956217960ull % MOD,
    11596043559919659130ull % MOD,
    2976841507452846995ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    15090073748392700862ull % MOD,
    3496786927732034743ull % MOD,
    8646735362535504000ull % MOD,
    2460088694130347125ull % MOD,
    3944675034557577794ull % MOD,
    14781700518249159275ull % MOD,
    2857749437648203959ull % MOD,
    8505429584078195973ull % MOD,
    18008150643764164736ull % MOD,
    720176627102578275ull % MOD,
    7038653538629322181ull % MOD,
    8849746187975356582ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    17427790390280348710ull % MOD,
    1159544160012040055ull % MOD,
    17946663256456930598ull % MOD,
    6338793524502945410ull % MOD,
    17715539080731926288ull % MOD,
    4208940652334891422ull % MOD,
    12386490721239135719ull % MOD,
    10010817080957769535ull % MOD,
    5566101162185411405ull % MOD,
    12520146553271266365ull % MOD,
    4972547404153988943ull % MOD,
    5597076522138709717ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,

    18338863478027005376ull % MOD,
    115128380230345639ull % MOD,
    4427489889653730058ull % MOD,
    10890727269603281956ull % MOD,
    7094492770210294530ull % MOD,
    7345573238864544283ull % MOD,
    6834103517673002336ull % MOD,
    14002814950696095900ull % MOD,
    15939230865809555943ull % MOD,
    12717309295554119359ull % MOD,
    4130723396860574906ull % MOD,
    7706153020203677238ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
    0ull % MOD,
};
