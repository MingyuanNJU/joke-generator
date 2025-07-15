// 1. 获取HTML中的元素
// 我们需要操作“显示笑话的p标签”和“换个笑话的按钮”
const jokeDisplay = document.getElementById('joke-display');
const newJokeBtn = document.getElementById('new-joke-btn');

// 2. 定义获取笑话的函数
// async/await 是处理异步操作（比如网络请求）的现代方式
async function getJoke() {
    // 在请求开始前，显示“加载中...”
    jokeDisplay.textContent = '加载中...';

    try {
        // API地址
        const apiUrl = 'https://v2.jokeapi.dev/joke/Any?type=single';

        // 3. 发送网络请求
        // await 会等待 fetch 操作完成
        const response = await fetch(apiUrl);
        // await 会等待 .json() 操作完成，它会把返回的数据解析成JS对象
        const data = await response.json();

        // 4. 检查请求是否成功并且有笑话内容
        if (data && data.joke) {
            // 如果成功，将笑话显示在p标签中
            jokeDisplay.textContent = data.joke;
        } else {
            // 如果API返回的数据格式不对
            jokeDisplay.textContent = '抱歉，获取笑话失败了！';
        }
    } catch (error) {
        // 5. 捕获可能发生的错误（比如网络断了）
        console.error('请求错误:', error);
        jokeDisplay.textContent = '网络出错了，请稍后再试！';
    }
}

// 6. 给按钮添加点击事件监听器
// 当按钮被点击时，调用 getJoke 函数
newJokeBtn.addEventListener('click', getJoke);

// 7. 页面首次加载时，也调用一次 getJoke 函数
// 这样一打开页面就有笑话了
getJoke();