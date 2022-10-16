// variables
let userName = null;
let state = 'SUCCESS';

//입력된 message를 창에 추가하여주는 생성자 
function Message(arg) {
	this.text = arg.text;
	this.message_side = arg.message_side;

	this.draw = function(_this) {
		return function() {
			let $message;
			$message = $($('.message_template').clone().html());
			$message.addClass(_this.message_side).find('.text').html(_this.text);
			$('.messages').append($message);

			return setTimeout(function() {
				return $message.addClass('appeared');
			}, 0);
		};
	}(this);
	return this;
}


//메세지 보내는 함수이고 'left'를 음성으로 변환한다 
function sendMessage(text, message_side) {
	if (message_side == 'left') speech(text)
	if (message_side == 'end') {
		startchat = false
		speech(text)
		message_side = 'left'
	}

	let $messages, message;
	$('.message_input').val('');
	$messages = $('.messages');
	message = new Message({
		text: text,
		message_side: message_side
	});
	message.draw();
	$messages.animate({ scrollTop: $messages.prop('scrollHeight') }, 300);
}


function greet() {
	setTimeout(function() {
		return sendMessage("EZ-ORDER 음성인식 주문을 시작합니다.", 'left');
	}, 1000);

	setTimeout(function() {
		return sendMessage("주문자의 성함을 알려주세요.", 'left');
	}, 2000);


}



//setUser name
function setUserName(username) {

	if (username != null && username.replace(" ", "" !== "")) {

		setTimeout(function() {

			return sendMessage("반갑습니다." + username + "님.", 'left');
		}, 1000);
		setTimeout(function() {

			return sendMessage("주문 할 메뉴를 선택하세요.", 'left');
		}, 2000);
		

		return username;

	} else {
		setTimeout(function() {
			return sendMessage("올바른 닉네임을 이용해주세요.", 'left');
		}, 1000);

		return null;
	}
}

// backend deeplearnning
function requestChat(messageText) {
	//ajax로 답변글 리턴 한다 
//	console.log(messageText)
//	return userName + "님의 " + messageText + " 답변입니다"   
    let url="/chat/"
    console.log(url)
    $.ajax({
        url:url ,
        type: "POST",
        dataType: "json",
        async: false,
        data : {'text':messageText},
        success: function (data) {
            console.log(data)        
            setTimeout(function () {
                return sendMessage(data['result'], 'left');
            }, 1000);
        },
        error: function (request, status, error) {
            console.log(error);
            
            return sendMessage('죄송합니다. 서버 연결에 실패했습니다.', 'left');
        }
    });


}


//onclick event
function onSendButtonClicked(speechMsg) {
	let messageText = speechMsg;
	if (speechMsg) sendMessage(messageText, 'right');
	console.log(speechMsg)
	if (userName == null) {
		userName = setUserName(messageText);

	} else {
		if (messageText.includes('안녕')) {
			setTimeout(function() {

				return sendMessage("안녕하세요. 저는 EZ-ORDER 입니다.", 'left');

			}, 1000);
		} else if (messageText.includes('고마워')) {
			setTimeout(function() {
				return sendMessage("천만에요. 더 필요하신 건 없나요?", 'left');
			}, 1000);
		} else if (messageText.includes('없어')) {
			setTimeout(function() {
				return sendMessage("그렇군요. 알겠습니다!", 'left');
			}, 1000);

		} else if (messageText.includes('끝', '종료')) {
			setTimeout(function() {
				startchat = false
				return sendMessage("네 주문을 종료합니다.", 'end');
			}, 1000);

		} else {
			let msg = requestChat(messageText, 'request_chat')
			console.log(msg)
			return 
		}

	}
}