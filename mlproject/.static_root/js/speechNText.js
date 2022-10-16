/* 1.음성인식  */
window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

const recognition = new SpeechRecognition();
recognition.interimResults = true;
recognition.lang = 'ko-KR';
let startchat = true;  //chat 시작 
let p = null;  //입력된 음성 text
recognition.onstart = function() {
  console.log("음성 입력을 시작 하였습니다")
};
recognition.onend = function() {
	if  (startchat)   // 음성 챗봇 시작중
	{
		recognition.start();
		if (p) onSendButtonClicked(p)
		p = '';
	}
};
recognition.onresult = function(e) {
	let speechMsg = Array.from(e.results)
		.map(results => results[0].transcript).join("");
	speechMsg.replace(/느낌표|강조|뿅/gi, '❗️');
	p = speechMsg    //입력된 음성 text
};

window.addEventListener("DOMContentLoaded", function(e) {
	//	var t = e.target;
	//	var input = t.previousElementSibling;
	greet();
	startchat = true;
	userName=null;
	recognition.start();
	speech('음성 주문을 시작합니다 ');
});


/* 2. text to speech  */
/* !!! 인위적인 이밴트를 발생시켜야만 실행됨  */
var voices = [];
setVoiceList();
if (window.speechSynthesis.onvoiceschanged !== undefined) {
	window.speechSynthesis.onvoiceschanged = setVoiceList;
}
function setVoiceList() {
	voices = window.speechSynthesis.getVoices();
}
function speech(txt) {
	if (!window.speechSynthesis) {
		alert("음성 재생을 지원하지 않는 브라우저입니다. 크롬, 파이어폭스 등의 최신 브라우저를 이용하세요");
		return;
	}
	var lang = 'ko-KR';
	var utterThis = new SpeechSynthesisUtterance(txt);
	utterThis.onend = function(event) {
		console.log('end');
	};
	utterThis.onerror = function(event) {
		console.log('error', event);
	};
	var voiceFound = false;
	for (var i = 0; i < voices.length; i++) {
		if (voices[i].lang.indexOf(lang) >= 0 || voices[i].lang.indexOf(lang.replace('-', '_')) >= 0) {
			utterThis.voice = voices[i];
			voiceFound = true;
		}
	}
	if (!voiceFound) {
		alert('voice not found');
		return;
	}
	utterThis.lang = lang;
	utterThis.pitch = 1;
	utterThis.rate = 1; //속도
	window.speechSynthesis.speak(utterThis);
}
